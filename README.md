# Dog Breed

A example of pytorch custom dataset.

## Processed
1. Read image ids and breeds from labels.csv.
2. Create breed dictionary based on breed information.
3. Append "image pathname" or "None" to labels.csv based on the results of if image exist or not.
4. Append breed ids to labels.csv, which are the mapping result of breed information and dictionary.
5.Save image pathanme and breed ids as npy file.

