%%bash

# Define the arrays of parameters
clustering_types=('agglomerative' 'kmeans')
metrics=('dtw' 'euclidean')
normalization_types=('no' 'z' 'min-max')

# Loop over each combination of parameters and run the Python script
for clustering_type in "${clustering_types[@]}"; do
  for metric in "${metrics[@]}"; do
    for normalization_type in "${normalization_types[@]}"; do
      output_file="results_${clustering_type}_${metric}_${normalization_type}.pkl"
      echo "Running experiment with clustering_type=${clustering_type}, metric=${metric}, normalization_type=${normalization_type}"
      python3 automate_clustering.py --type "$clustering_type" --norm "$normalization_type" --metric "$metric" --start 10 --end 400 --step 30 \
                                     --input './data/team_phases/Manchester City.csv' \
                                     --output "$output_file"
    done
  done
done
