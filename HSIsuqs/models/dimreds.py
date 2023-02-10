import numpy as np
from sklearn.decomposition import PCA


def img_pca(img, n_dims=None, percent_var=.9, seed=420, pca_args={}):
    data = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

    pca = PCA(n_components=n_dims, random_state=seed, **pca_args)
    new_data = pca.fit_transform(data)
    pca.explained_variance_ratio_sum = np.cumsum(pca.explained_variance_ratio_)

    if n_dims is None:
        n_dims = np.argwhere(pca.explained_variance_ratio_sum > percent_var)
        n_dims = n_dims[0][0]
    new_data = new_data[:, :n_dims]
    new_img = new_data.reshape(img.shape[0], img.shape[1], n_dims)

    return {'new_img': new_img,
            'pca': pca,
            'n_dims': n_dims}

def estimate_n_segs_pca(img, percent_var, img_pca_args={}):
    res = img_pca(img, None, percent_var, **img_pca_args)
    return res['n_dims']



