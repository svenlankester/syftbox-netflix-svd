import os
import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_top5_data(recommendation_list, df):
    top5_data = []
    for i, (name, idx, score) in enumerate(recommendation_list):
        print(f"\t{i+1} => {name}: {score:.4f}")

        # Locate the row in the DataFrame with the matching title
        row = df[df["Title"].str.strip() == name].iloc[0]
        entry = {
            "id": int(idx),  # Use the index from `top5_indices`
            "name": name,
            "language": row["Language"],
            "rating": row["Rating"],
            "imdb": row["IMDB"] if pd.notna(row["IMDB"]) else "N/A",  # Default to N/A if IMDB is missing
            "img": row["Cover URL"],
            "count": int(score),
            "raw_score": score
        }
        top5_data.append(entry)
    return top5_data

def local_recommendation(local_path, global_path, tv_vocab, exclude_watched=True):
    """Main entry point for local recommendation generation."""
    from syftbox_netflix.federated_learning.svd_participant_local_recommendation import compute_recommendations

    # Step 1: Load data
    global_V_path = os.path.join(global_path, "global_V.npy")
    user_U_path = os.path.join(local_path, "svd_training", "U.npy")
    user_aggregated_activity_path = os.path.join(local_path, "netflix_aggregated.npy")
    
    if not os.path.exists(user_U_path) or not os.path.exists(user_aggregated_activity_path):
        logging.error(f"User data not found. Exiting recommendation process. --> U.npy ({user_U_path}) / netflix_aggregated.npy ({user_aggregated_activity_path})")
        return

    user_U = np.load(user_U_path)
    global_V = np.load(global_V_path)
    user_aggregated_activity = np.load(user_aggregated_activity_path)

    # Step 2: Run process to compute recommendations
    raw_recommendations, reranked_recommendations = compute_recommendations(user_U, global_V, tv_vocab, user_aggregated_activity, exclude_watched=exclude_watched)

    # Step 3: Write or return recommendations
    # Enhance data
    csv_file_path = os.path.join(os.getcwd(), "syftbox_netflix", "aggregator", "data", "netflix_series_2024-12.csv.zip")
    try:
        df = pd.read_csv(csv_file_path, compression='zip')
    except Exception as e:
        print(f"> Error: Unable to read the CSV from {csv_file_path}. Error: {e}")
        return

    print("(Unprocessed) Recommended based on most recently watched:")
    raw_top5 = get_top5_data(raw_recommendations, df)
    print("(Re-ranked) Recommended based on most recently watched:")
    reranked_top5 = get_top5_data(reranked_recommendations, df) 

    return raw_top5, reranked_top5
