import torch
import tempfile
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from .vae import LinearVAE
import numpy as np
from onpolicy.utils.shared_buffer import SharedReplayBuffer

# Based on https://github.com/uoe-agents/seps

class rbDataSet(Dataset):
    def __init__(self, rb:SharedReplayBuffer, encoder_in=['agent_id'], decoder_in=['obs', 'action'], reconstruct=["next_obs", "reward"]):
        self.rb = rb
        
        # All features to shape (num_env_steps * num_envs * num_agents, dim)
        num_env_steps, num_envs, num_agents = rb.actions.shape[:3]
        # create one hot encodeing for agent ids
        agent_id = np.zeros((num_env_steps, num_envs, num_agents, num_agents))
        for i in range(num_agents):
            agent_id[:, :, i, i] = 1
        agent_id = agent_id.reshape(-1, num_agents)
        obs = rb.obs[:-1].reshape(-1, *rb.obs.shape[3:])
        next_obs = rb.obs[1:].reshape(-1, *rb.obs.shape[3:])
        actions = rb.actions.reshape(-1, *rb.actions.shape[3:])
        rewards = rb.rewards.reshape(-1, *rb.rewards.shape[3:])

        encoder_in = [agent_id]
        decoder_in = [obs, actions]
        reconstruct = [next_obs]
        self.data = []
        self.data.append(torch.cat([torch.from_numpy(n) for n in encoder_in], dim=1).float())
        self.data.append(torch.cat([torch.from_numpy(n) for n in decoder_in], dim=1).float())
        self.data.append(torch.cat([torch.from_numpy(n) for n in reconstruct], dim=1).float())

        # normalize reconstruct features to make the loss more interpretable
        self.data[2] = 2 * (self.data[2]-self.data[2].min())/(self.data[2].max()-self.data[2].min()) - 1
        
        print([x.shape for x in self.data])

    def __len__(self):
        return self.data[0].shape[0]
    def __getitem__(self, idx):
        return [x[idx, :] for x in self.data]

def find_optimal_cluster_number(X):
    from sklearn.metrics import silhouette_score
    range_n_clusters = list(range(2, X.shape[0]))
    silhouette_scores = []
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        s_score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(s_score)

    print('sil:', range_n_clusters[np.argmax(silhouette_scores)])
    max_key = range_n_clusters[np.argmax(silhouette_scores)] 

    return max_key

def compute_clusters(buffer, num_agents, batch_size=128, num_clusters=None, lr=3e-4, epochs=10, z_features=10, kl_weight=0.0001, visualize=False):
    device = "cuda"

    dataset = rbDataSet(buffer)
    
    input_size = dataset.data[0].shape[-1]
    extra_decoder_input = dataset.data[1].shape[-1]
    reconstruct_size = dataset.data[2].shape[-1]
    
    model = LinearVAE(z_features, input_size, extra_decoder_input, reconstruct_size)
    print(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # criterion = nn.BCELoss(reduction='sum')
    criterion = nn.MSELoss(reduction="sum")
    def final_loss(bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the 
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        BCE = bce_loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + kl_weight*KLD

    def fit(model, dataloader):
        model.train()
        running_loss = 0.0
        for i, (encoder_in, decoder_in, y) in enumerate(dataloader):
            encoder_in = encoder_in.to(device)
            decoder_in = decoder_in.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(encoder_in, decoder_in)
            # print(torch.cat([reconstruction.unsqueeze(-1)[0], y.unsqueeze(-1)[0]], dim=-1))
            bce_loss = criterion(reconstruction, y)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = running_loss/len(dataloader.dataset)
        return train_loss

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loss = []
    for epoch in tqdm(range(epochs)):
        train_epoch_loss = fit(model, dataloader)
        train_loss.append(train_epoch_loss)
        # print(f'epoch {epoch} - loss {train_epoch_loss/dataset.data[2].shape[-1]}')

    print(f"Train Loss: {train_epoch_loss:.6f}")
    x = torch.eye(num_agents).to(device)

    with torch.no_grad():
        z = model.encode(x)
    z = z.to("cpu")
    z = z[:, :]

    if num_clusters is None:
        num_clusters = find_optimal_cluster_number(z)
    print(f"Creating {num_clusters} clusters.")
    # run k-means from scikit-learn
    kmeans = KMeans(
        n_clusters=num_clusters, init='k-means++',
        n_init=10,
        random_state=42
    )
    cluster_ids_x = kmeans.fit_predict(z) # predict labels
    if visualize:
        visualize_clusters(z, cluster_ids_x)

    return cluster_ids_x.tolist()

def plot_clusters(cluster_centers, z, human_selected_idx=None):

    if human_selected_idx is None:
        plt.plot(z[:, 0], z[:, 1], 'o')
        plt.plot(cluster_centers[:, 0], cluster_centers[:, 1], 'x')

        for i in range(z.shape[0]):
            plt.annotate(str(i), xy=(z[i, 0], z[i, 1]))

    else:
        colors = 'bgrcmykw'
        for i in range(len(human_selected_idx)):
            plt.plot(z[i, 0], z[i, 1], 'o' + colors[human_selected_idx[i]])

        plt.plot(cluster_centers[:, 0], cluster_centers[:, 1], 'x')
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile, format="png") # File position is at the end of the file.

def visualize_clusters(data, cluster_ids):
    if data.shape[1] > 2:
        # from sklearn.manifold import TSNE
        # tsne = TSNE(n_components=2, random_state=42, perplexity=2)
        # reduced_data = tsne.fit_transform(data)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
    else:
        reduced_data = data

    plt.figure(figsize=(8, 6))
    for id in np.unique(cluster_ids):
        cluster_points = reduced_data[cluster_ids == id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {id}")

    # Add plot details
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title('silhouette_score')
    plt.legend()
    plt.savefig("cluster_img/cluster_data_sil.png", dpi=300)
    plt.close()