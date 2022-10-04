from train import train
import pandas as pd
import itertools

if __name__ == '__main__':
    all_emb_dims = [100, 200, 300]
    all_h_dims = [100, 200, 300]
    all_latent_dims = [15, 30]
    all_activations = ['gumbel', 'bc']

    all_results = []
    i = 0
    for emb_dim, h_dim, latent_dim, activation in itertools.product(all_emb_dims, all_h_dims, all_latent_dims,
                                                                    all_activations):
        train_id = f'tunning_{i}'
        result_df = train(train_id=train_id, emb_dim=emb_dim, h_dim=h_dim, latent_dim=latent_dim, epochs=50,
                          activation=activation)
        all_results.append(result_df)
        i += 1

    tunning_df = pd.concat(all_results)
    tunning_df.to_csv("train_results/tunning_results.csv", index=False)
