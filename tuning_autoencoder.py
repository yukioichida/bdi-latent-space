from train import train
import pandas as pd
import itertools

if __name__ == '__main__':
    all_emb_dims = [50, 100, 150]
    all_h_dims = [50, 100, 150]
    all_latent_dims = [15, 30, 45]

    all_results = []
    for emb_dim, h_dim, latent_dim in itertools.product(all_latent_dims, all_h_dims, all_latent_dims):
        result_df = train(emb_dim=emb_dim, h_dim=h_dim, latent_dim=latent_dim, epochs=70)
        all_results.append(result_df)

    tunning_df = pd.concat(all_results)
    tunning_df.to_csv("results/tunning_results.csv")
