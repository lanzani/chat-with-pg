from sklearn.decomposition import PCA
import plotly.express as px
from langchain_chroma import Chroma

from populate_db import CHROMA_PATH
from utils import get_embedding_function

# Get embeddings
embedding_function = get_embedding_function()
chroma = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

collection = chroma._client.get_collection("essays")
embeddings = collection.get(include=['embeddings'])['embeddings']
docs = collection.get(include=['embeddings'])['ids']


# Reduce the embedding dimensionality
pca = PCA(n_components=3)
vis_dims = pca.fit_transform(embeddings)# Create an interactive 3D plot
fig = px.scatter_3d(
    x=vis_dims[:, 0],
    y=vis_dims[:, 1],
    z=vis_dims[:, 2],
    text=docs,
    labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'z': 'PCA Component 3'}, # Name it like you want
    title='3D PCA of Embeddings' # Name it like you want
)

fig.show()