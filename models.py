import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from transformers import AutoImageProcessor, Dinov2Model, CLIPSegProcessor, CLIPSegForImageSegmentation, CLIPSegModel
import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from segment_anything import sam_model_registry, SamPredictor
from deepcrack_config import Config as cfg
from deepcrack_checkpointer import Checkpointer

class EfficientNetB3FeatureExtractor:
    def __init__(self):
        self.model = EfficientNetB3(weights='imagenet', include_top=False)

    def extract_features(self, img_array):
        features = self.model.predict(img_array)
        return features

class DinoV2FeatureExtractor:
    def __init__(self, modelpath):
        self.image_processor = AutoImageProcessor.from_pretrained(modelpath)
        self.model = Dinov2Model.from_pretrained(modelpath)

    def extract_features(self, img_array):
        inputs = self.image_processor(img_array, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        output = outputs.last_hidden_state
        return output

class CLIPSegForImageSegmentationFeatureExtractor:
    def __init__(self, modelpath):
        self.image_processor = CLIPSegProcessor.from_pretrained(modelpath, local_files_only=True)
        self.model = CLIPSegForImageSegmentation.from_pretrained(modelpath, local_files_only=True)

    def extract_features(self, img_array, prompt):
        inputs = self.image_processor(text=prompt, images=[img_array]*len(prompt), padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        output = torch.stack(outputs.decoder_output.hidden_states)
        return output

class CLIPSegModelFeatureExtractor:
    def __init__(self, modelpath):
        self.image_processor = CLIPSegProcessor.from_pretrained(modelpath, local_files_only=True)
        self.model = CLIPSegModel.from_pretrained(modelpath, local_files_only=True)

    def extract_features(self, img_array):
        inputs = self.image_processor(images=img_array, return_tensors="pt")
        output = self.model.get_image_features(**inputs)
        return output

class SAMFeatureExtractor:
    def __init__(self, modelpath):
        self.model = sam_model_registry['vit_b'](checkpoint=modelpath)
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.predictor = SamPredictor(self.model)

    def extract_features(self, img_array):
        self.predictor.set_image(img_array)
        output = self.predictor.get_image_embedding()
        return output

class UnetPPModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(arch, encoder_name=encoder_name, in_channels=in_channels,classes=out_classes, **kwargs)
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1,3,1,1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1,3,1,1))

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        return self.model(image.float())

    def shared_step(self, batch, stage):
        image = batch["image"]
        mask = batch["mask"]

        logits_mask = self.forward(image)

        loss = self.loss_fn(logits_mask, mask)

        if stage == "valid":
            self.validation_step_outputs.append(loss)
        elif stage == "train":
            self.training_step_outputs.append(loss)
        else:
            self.test_step_outputs.append(loss)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average, sync_dist=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "valid")
        self.log("val_loss", loss, sync_dist=True)
        return loss
    
    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", epoch_average, sync_dist=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")
    
    def on_test_epoch_end(self):
        epoch_average = torch.stack(self.test_step_outputs).mean()
        self.log("test_epoch_average", epoch_average, sync_dist=True)
        self.test_step_outputs.clear()
    
    def configure_parameters(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


# Deepcrack classes -- https://github.com/qinnzou/DeepCrack/

def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv3X3(in_, out)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class Down(nn.Module):

    def __init__(self, nn):
        super(Down,self).__init__()
        self.nn = nn
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self,inputs):
        down = self.nn(inputs)
        unpooled_shape = down.size()
        outputs, indices = self.maxpool_with_argmax(down)
        return outputs, down, indices, unpooled_shape

class Up(nn.Module):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        self.unpool=torch.nn.MaxUnpool2d(2,2)

    def forward(self,inputs,indices,output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        outputs = self.nn(outputs)
        return outputs

class Fuse(nn.Module):

    def __init__(self, nn, scale):
        super().__init__()
        self.nn = nn
        self.scale = scale
        self.conv = Conv3X3(64,1)

    def forward(self,down_inp,up_inp):
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = F.interpolate(outputs, scale_factor=self.scale, mode='bilinear')
        outputs = self.nn(outputs)

        return self.conv(outputs)



class DeepCrack(nn.Module):

    def __init__(self, num_classes=1000):
        super(DeepCrack, self).__init__()

        self.down1 = Down(torch.nn.Sequential(
            ConvRelu(3,64),
            ConvRelu(64,64),
        ))

        self.down2 = Down(torch.nn.Sequential(
            ConvRelu(64,128),
            ConvRelu(128,128),
        ))

        self.down3 = Down(torch.nn.Sequential(
            ConvRelu(128,256),
            ConvRelu(256,256),
            ConvRelu(256,256),
        ))

        self.down4 = Down(torch.nn.Sequential(
            ConvRelu(256, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 512),
        ))

        self.down5 = Down(torch.nn.Sequential(
            ConvRelu(512, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 512),
        ))

        self.up1 = Up(torch.nn.Sequential(
            ConvRelu(64, 64),
            ConvRelu(64, 64),
        ))

        self.up2 = Up(torch.nn.Sequential(
            ConvRelu(128, 128),
            ConvRelu(128, 64),
        ))

        self.up3 = Up(torch.nn.Sequential(
            ConvRelu(256, 256),
            ConvRelu(256, 256),
            ConvRelu(256, 128),
        ))

        self.up4 = Up(torch.nn.Sequential(
            ConvRelu(512, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 256),
        ))

        self.up5 = Up(torch.nn.Sequential(
            ConvRelu(512, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 512),
        ))

        self.fuse5 = Fuse(ConvRelu(512 + 512, 64), scale=16)
        self.fuse4 = Fuse(ConvRelu(512 + 256, 64), scale=8)
        self.fuse3 = Fuse(ConvRelu(256 + 128, 64), scale=4)
        self.fuse2 = Fuse(ConvRelu(128 + 64, 64), scale=2)
        self.fuse1 = Fuse(ConvRelu(64 + 64, 64), scale=1)

        self.final = Conv3X3(5,1)

    def forward(self,inputs):

        # encoder part
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)
        out, down5, indices_5, unpool_shape5 = self.down5(out)

        # decoder part
        up5 = self.up5(out, indices=indices_5, output_shape=unpool_shape5)
        up4 = self.up4(up5, indices=indices_4, output_shape=unpool_shape4)
        up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)

        fuse5 = self.fuse5(down_inp=down5,up_inp=up5)
        fuse4 = self.fuse4(down_inp=down4, up_inp=up4)
        fuse3 = self.fuse3(down_inp=down3, up_inp=up3)
        fuse2 = self.fuse2(down_inp=down2, up_inp=up2)
        fuse1 = self.fuse1(down_inp=down1, up_inp=up1)

        output = self.final(torch.cat([fuse5,fuse4,fuse3,fuse2,fuse1],1))

        return output, fuse5, fuse4, fuse3, fuse2, fuse1

def get_optimizer(model):
    if cfg.use_adam:
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum, )

class DeepCrackTrainer(nn.Module):
    def __init__(self, model):
        super(DeepCrackTrainer, self).__init__()
        self.model = model

        self.saver = Checkpointer(cfg.name, cfg.saver_path, overwrite=False, verbose=True, timestamp=True,
                                  max_queue=cfg.max_save)

        self.optimizer = get_optimizer(self.model)

        self.iter_counter = 0

        # -------------------- Loss --------------------- #

        self.mask_loss = nn.BCEWithLogitsLoss(reduction='mean',
                                              pos_weight=torch.FloatTensor([cfg.pos_pixel_weight]))

        self.log_loss = {}
        self.log_acc = {}