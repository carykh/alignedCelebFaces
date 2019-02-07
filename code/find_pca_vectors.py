import numpy as np
from sklearn.decomposition import PCA




denseRep = np.load("denseArray27K.npy")
norm_dense_rep = denseRep-np.mean(denseRep, axis = 0)

'''cov_mat=np.cov(norm_dense_rep, rowvar=False)

values, vectors = np.linalg.eigh(cov_mat)
indices = np.argsort(values)[::-1]
values = values[indices]
vectors = vectors[:,indices]
np.save("eigenvalues.npy",values)
np.save("eigenvectors.npy",vectors)'''


pca = PCA(n_components=norm_dense_rep.shape[1])
pca.fit(norm_dense_rep)
values = np.sqrt(pca.explained_variance_)
vectors = pca.components_
np.save("eigenvalues.npy",values)
np.save("eigenvectors.npy",vectors)

