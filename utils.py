import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA

def reduce_dimensions(outputs, names, n_components, feature_extractor, dataset_name):
    # reduce dimensions of the outputs
    transformer = PCA(n_components=n_components)
    data_nd = transformer.fit_transform(np.array(outputs))
    print(n_components, len(data_nd[0]))
    assert n_components == len(data_nd[0])

    primary_names, secondary_names = names
    primary_points = data_nd[:len(primary_names)]
    secondary_points = data_nd[len(primary_names):]

    scores = {}

    for dim_idx in range(5, n_components, 5):
        pairwise_secondary = scipy.spatial.distance.cdist(secondary_points[:,:dim_idx], secondary_points[:, :dim_idx])

        pairwise_secondary_primary = scipy.spatial.distance.cdist(secondary_points[:,:dim_idx], primary_points[:, :dim_idx])

        secondary_distances = np.sum(pairwise_secondary_primary, axis=-1)

        secondary_names = names[len(primary_points): ]

        idist_map = {}
        for dist, name in zip(secondary_distances, secondary_names):
            idist_map[name] = dist
        np.save(f"idist/{feature_extractor}_distances_{dataset_name}_dim_{dim_idx}.npy", idist_map)


        scores[f"PC-{dim_idx}"] = np.mean(secondary_distances)

    return scores