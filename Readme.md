### Environment & Data Setup

Before building the model, we need to configure our notebook environment for high-performance processing and load our data.

#### 1. Data Configuration
* **Kaggle Datasets**: All project data (the `train.csv`, `test.csv`, and the `images/` directory) was first uploaded as a private Kaggle Dataset.
* **Notebook Import**: We then imported this dataset into the notebook. This is the most efficient way to handle large files on Kaggle, as the data is mounted directly to the notebook's filesystem under the `/kaggle/input/` directory.

#### 2. Accelerator: GPU T4 x2
* **Enabled GPU**: To dramatically speed up our feature extraction, we enabled a high-performance GPU accelerator. This can be done in the notebook's right-hand menu:
    `Settings` -> `Accelerator` -> `GPU T4 x2`
* **Why T4 x2?**: This option provides us with **two** NVIDIA T4 GPUs. For an "embarrassingly parallel" task like `model.predict()`, TensorFlow can automatically split the prediction workload across both GPUs, effectively **halving the time** required for feature extraction.

---

### Load Image Data using `tf.data`

We use the `tf.keras.utils.image_dataset_from_directory` utility to create a `tf.data.Dataset` object. This is a highly efficient way to load images from a directory directly into a TensorFlow data pipeline.

* `root_dir`: The path to our images.
* `labels=None` / `label_mode=None`: We explicitly tell the function *not* to look for labels. This is because our images are not organized into subfolders by class (e.g., `/dogs`, `/cats`). We will get our labels from a separate CSV file.
* `batch_size=64`: Loads the images in batches of 64.
* `image_size=(224, 224)`: Resizes all images to 224x224 to match the input shape required by our MobileNetV2 model.
* `shuffle=False`: This is **critical**. We must not shuffle the dataset here so that the order of the images loaded from the directory perfectly matches the order of the corresponding data (like labels or IDs) in our `train.csv` or `test.csv` file.
* `interpolation='lanczos5'`: Uses a high-quality resampling filter for resizing the images.

Finally, we apply `.prefetch(buffer_size=tf.data.AUTOTUNE)`. This is a performance optimization that allows the CPU to pre-load the next batch of images while the GPU is busy processing the current one, preventing data bottlenecks.

---

### Load Pre-trained Base Model (MobileNetV2)

We initialize the **MobileNetV2** model, pre-trained on the ImageNet dataset.

* `input_shape=(224, 224, 3)`: Specifies the input size for our images (224x224 pixels with 3 color channels).
* `include_top=False`: Excludes the final fully-connected (classification) layer from the original MobileNetV2. This allows us to add our own custom classifier suited for our specific task.
* `weights='imagenet'`: Loads weights pre-trained on the ImageNet dataset, which is crucial for transfer learning.
* `pooling='avg'`: Applies **Global Average Pooling** to the output of the base model. This converts the feature maps into a single flat vector per image, making it easy to connect to our new classification layer.

---

### Debugging: Finding Corrupted Images (Attempt 1)

During the feature extraction, the `model.predict()` process failed at a specific batch (`Batch 1009`). This almost always indicates that one of the image files in that batch is corrupted or truncated. This script is a targeted debugger to find the exact problematic file without having to re-scan the entire dataset from the beginning.

Here is the logic:
1.  **Configuration**: We set the `FAILED_AT_BATCH` variable to the batch number where the error occurred (1009) and note the `BATCH_SIZE` (64).
2.  **Calculate Offset**: We calculate a `start_index` to skip all files from the batches that we know completed successfully (i.e., batches 1 through 1008).
3.  **Gather & Sort Files**: The script gets a complete list of all image paths from the directory.
4.  **Critical Sort**: It performs an alphabetical sort on the file list. This is **essential** to ensure the list's order perfectly matches the order used by `tf.keras.utils.image_dataset_from_directory` (when `shuffle=False`).
5.  **Targeted Scan**: It creates a new, smaller list (`files_to_check`) containing only the "suspect" files, starting from the calculated `start_index`.
6.  **Verify with TensorFlow**: It iterates *only* through this suspect list and attempts to open and decode each file using TensorFlow's own I/O functions.

---

### Debugging & Cleaning: Finding All Corrupted Images (Attempt 2)

This script is an optimized debugger designed to find and handle *all* problematic files from the point of failure onward.

#### 1. Optimized Scanning
Instead of re-checking the entire dataset, we first calculate a `start_index`. This index tells the script to skip all the files from the batches that we *know* processed successfully.

#### 2. File Collection and Sorting
* The script gathers all image paths from the directory.
* **Critically**, it sorts the file list alphabetically (`all_files.sort()`). This is essential because it exactly matches the order that `tf.keras.utils.image_dataset_from_directory(shuffle=False)` uses to read the files.
* It then slices this main list to create a smaller `files_to_check` list, containing only the "suspect" files from the point of failure.

#### 3. TensorFlow-based Verification
The script loops through the suspect list and uses `tf.io.decode_image(..., channels=3)` to test each file. This is the most reliable method, as it uses the *exact* same decoding function that the TensorFlow model pipeline uses.

#### 4. Collect & Remove
* It finds and collects *all* corrupted files in the suspect range.
* It prints a final summary list of all bad files found.
* Finally, it attempts to `os.remove()` them. The `try...except` block correctly anticipates and handles the `OSError` (read-only filesystem error), which is expected when working in the `/kaggle/input/` directory.

---

### Building a Custom `tf.data` Pipeline to Skip Corrupted Files

Our previous debugging step successfully identified corrupted files. Since we cannot remove these files from the read-only `/kaggle/input/` directory, we must build a new data pipeline that manually filters them out *before* processing. This code replaces the original `image_dataset_from_directory` with a more flexible `tf.data` pipeline.

#### Step 1: Filter Out Corrupted Files
First, we define a "blocklist" (`corrupted_files_list`) containing the paths of all known bad files.
* We convert this list to a `set` for high-performance lookups.
* We get a list of *all* file paths in the directory.
* We create our final `good_file_paths` list by iterating through all files and keeping only the ones *not* present in the `corrupted_files_set`.
* Finally, we **sort** this clean list to ensure the image order remains consistent.

#### Step 2: Build the Custom `tf.data` Pipeline
* `tf.data.Dataset.from_tensor_slices`: We create a new dataset directly from our clean list of `good_file_paths`.
* `load_and_preprocess_image`: We define a helper function that does the work `image_dataset_from_directory` used to do for us:
    1.  `tf.io.read_file`: Reads the image file from its path.
    2.  `tf.io.decode_jpeg`: Decodes the raw bytes into an image tensor.
    3.  `tf.image.resize`: Resizes the image to `(224, 224)`.
* We then apply this function using `.map()` and add the standard performance optimizations: `.batch()` and `.prefetch()`.

#### Step 3: Define Model and Extract Features
* We define the exact same `MobileNetV2` `base_model` as before.
* We call `base_model.predict()`, but this time we pass it our new, custom, and **guaranteed-clean** `image_dataset`.

---

### Extract Features (Generate Embeddings)

We now pass our entire `image_dataset` through the pre-trained `base_model` using the `.predict()` method.

* This is the main step of **feature extraction**.
* Since we set `include_top=False` and `pooling='avg'`, the model converts each image into a high-level feature vector (an "embedding").
* For each image in the dataset, the output will be a 1D vector of **1280 features**.

The `extracted_features` variable will now hold a large NumPy array with the shape `(total_number_of_images, 1280)`.

---

### Save Extracted Features (Checkpoint)

This is one of the most important steps for an efficient workflow. The `base_model.predict()` step was the most computationally expensive part of the notebook. We **do not** want to re-run this step every time.

By using `np.save()`, we save the resulting `extracted_features` array directly to disk as a `.npy` file.

* `np.save(file_path, extracted_features)`: This command takes our large feature array and writes it to the file `mobilenet_features.npy`.
* `np.load(file_path)`: This command will be used in all future sessions. It reloads the entire array from the file back into a variable almost instantly.

This creates a crucial **checkpoint**, allowing us to skip the expensive feature extraction and jump straight to training our final model.

---

### Text Feature Engineering: TF-IDF

To use the `catalog_content` text data, we must convert it into a numerical format using **TF-IDF**.

* `TfidfVectorizer(max_features=5000)`: We initialize the vectorizer, limiting the vocabulary to the **top 5,000 most frequent words**. This controls dimensionality and focuses on relevant terms.

* `X_text_sparse = vectorizer.fit_transform(train.catalog_content)`:
    * **fit**: The vectorizer "learns" the 5,000-word vocabulary from the training text.
    * **transform**: It then converts this training text into a **sparse matrix** (`X_text_sparse`), where each row is a document and each column is a word.

* `test_sparse = vectorizer.transform(test.catalog_content)`:
    * We *only* call `.transform()` on the test data. This is **critical** to apply the *same vocabulary* learned from the training data, ensuring the features are consistent.

---

### Convert Sparse Matrix to Dense Array

This line converts our `X_text_sparse` matrix into a standard, **dense** NumPy array by filling in all the un-stored values with zeros.

`np_arr_x_sparse = X_text_sparse.toarray()`

The resulting `np_arr_x_sparse` will have the shape `(number_of_documents, 5000)`.

---
**Important Note on Memory:**
Be very cautious with this operation. If the sparse matrix is large, the dense array will require a huge amount of RAM. Only use `.toarray()` if the model (like `tf.keras.Dense`) *requires* it. Models like XGBoost or LightGBM should use the sparse matrix directly.

---

### Save Sparse Matrix to Disk

We are saving our processed sparse text features (`X_test_sparse_test`) to a file using `scipy.sparse.save_npz`.

This is a crucial step for efficiency:

1.  **Saves Space**: It preserves the **sparse format**, which takes up significantly less disk space.
2.  **Saves Time**: It allows us to **reload** this processed data instantly in the future using `load_npz()`, skipping the time-consuming `TfidfVectorizer` step.

---

### Re-associating Image Features with Training Data

We need to map the features (calculated per *unique* image) back to our original `train` DataFrame (which has duplicate `image_link` entries).

#### Step 1: Create a Feature "Lookup" DataFrame

We first create a new, small DataFrame (`features_df`) that acts as a simple key-value "lookup table."

* `'image_link': sorted_unique_links`: We use our clean, sorted list of unique image links. This list **must** be in the exact same order as the feature vectors in `extracted_features`.
* `'mobilenet_features': list(extracted_features)`: We add our NumPy array of features.

This `features_df` now has one row for each unique image, linking its `image_link` to its `mobilenet_features`.

#### Step 2: Merge Back to the Original DataFrame

We use `pd.merge()` to combine our original `train` DataFrame with our new `features_df` lookup table.

* `on='image_link'`: Specifies the `image_link` column as the join key.
* `how='left'`: This is the most important part. It keeps **all** rows from the `train` data and attaches the corresponding `mobilenet_features` from the lookup table.

The `final_df` now has the original number of rows, but includes the new `mobilenet_features` column with the correct vector for each row.

---

### Final Feature Engineering: Expanding Image Features

Our `mobilenet_features` column is currently "packed" – each cell contains a vector of 1280 numbers. To make these features usable for models like XGBoost, we must "unpack" them so each of the 1280 features gets its **own column**.

#### Step 1: Expand the Feature Column

`expanded_features = concat_df['mobilenet_features'].apply(pd.Series)`

We use `.apply(pd.Series)` on the `mobilenet_features` column. This powerful function takes each list-like item and explodes it horizontally into a new DataFrame with 1280 columns (named `0`, `1`, `2`, ...).

#### Step 2: Rename New Columns

`expanded_features.columns = [f'f_{i}' for i in expanded_features.columns]`

We rename the new columns from integers to descriptive strings like `f_0`, `f_1`, `f_2`...

#### Step 3: Concatenate Back to the Main DataFrame

`full_final = pd.concat([..., expanded_features], axis=1)`

Finally, we combine our original data with our new expanded features.
1.  We **drop** the original "packed" `mobilenet_features` column.
2.  We `pd.concat` the `expanded_features` DataFrame side-by-side (`axis=1`).

The `full_final` DataFrame is now our complete, "flat" dataset, ready for modeling.

---

### Convert Final DataFrame to Sparse Matrix (CSR)

This is a critical **memory optimization** step. Our `full_final` DataFrame is currently **dense**, and all the zeros from our text and image features are wasting memory.

`full_final_csr = csr_matrix(full_final)`:

* **What it does**: This converts our entire dense DataFrame into a **`csr_matrix`** (Compressed Sparse Row) format.
* **How it works**: A CSR matrix **only stores the non-zero values** and their row/column coordinates.
* **Why it's important**:
    1.  **Memory**: The `full_final_csr` object will use a tiny fraction of the RAM.
    2.  **Speed**: Many models (**XGBoost, LightGBM**) are highly optimized to train directly and much faster on sparse matrices.

This makes it possible to train on our very high-dimensional dataset without running out of memory.

---

### Evaluate Model Performance (SMAPE)

Now we evaluate our trained `xgb_regressor` on the hold-out validation set (`X_eval`).

#### 1. Generate Predictions
`y_pred = xgb_regressor.predict(X_eval)`

#### 2. Define the Evaluation Metric: SMAPE
We must define the **SMAPE** (Symmetric Mean Absolute Percentage Error) function, as it is not built-in.

The formula is: $SMAPE = \frac{100}{n} \sum_{i=1}^{n} \frac{2 \cdot |F_i - A_i|}{|A_i| + |F_i|}$

* `epsilon = np.finfo(float).eps`: This is a critical addition. `epsilon` is the smallest possible positive number. We add it to the denominator to prevent a **divide-by-zero error** if both the actual and predicted value are 0.

#### 3. Calculate and Print the Final Score
Finally, we call our custom `smape` function to compare the true validation labels (`y_eval`) against our model's predictions (`y_pred`).

`print("SMAPE:", smape(y_eval, y_pred))`


## Save the Trained Model

This is the final and one of the most important steps of our training pipeline. We've spent a lot of time and computational resources training our `xgb_regressor` model. We must now **save (or "serialize") this trained object** to disk so we can use it later without retraining.

* **Why `joblib`?**: We use `joblib.dump()` because it is highly efficient for saving Python objects that contain large NumPy arrays, which is exactly what our trained XGBoost model is. It's generally preferred over other methods like `pickle` for scikit-learn-compatible models.

* **What it does**: The command saves the entire model—including all its learned internal parameters, feature importances, and configuration—into a single file named `xgbregressor_model.joblib`.

* **Next Steps**: This file is our final "artifact." We can now download it, or (more commonly) create a new, separate "Inference Notebook." In that notebook, we will simply load this file using `joblib.load()` and use it to generate predictions on the *actual competition test set*.
