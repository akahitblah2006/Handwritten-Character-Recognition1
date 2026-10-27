
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Step 1: Load dataset
digits = datasets.load_digits()

X = digits.data
y = digits.target

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 4: Check accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy * 100, "%")

# Step 5: Test model with a sample image
prediction = model.predict([X_test[0]])[0]
print("Test Image Prediction:", prediction)

plt.imshow(X_test[0].reshape(8, 8), cmap="gray")
plt.title(f"Test Image Prediction: {prediction}")
plt.show()

# Step 6: Load your handwritten image
file_path = r"C:\Users\Akshit\OneDrive\Desktop\seven.png"

try:
    # Open image in grayscale
    img = Image.open(file_path).convert("L")

    # Convert image to NumPy array
    img_array = np.array(img)

    # Convert white background to black if needed
    if img_array.mean() > 127:
        img_array = 255 - img_array

    # Find the boundaries of the handwritten digit
    rows, cols = np.where(img_array > 50)

    if len(rows) == 0:
        print("No digit found! Please check your image.")
    else:
        # Crop extra background
        img_array = img_array[
            rows.min():rows.max() + 1,
            cols.min():cols.max() + 1
        ]

        # Resize the digit
        digit_img = Image.fromarray(img_array).resize((6, 6))

        # Place digit in the center of an 8x8 image
        final_img = Image.new("L", (8, 8), 0)
        final_img.paste(digit_img, (1, 1))

        # Convert pixel values to match the dataset
        img_array = np.array(final_img)
        img_array = (img_array / 255.0) * 16

        # Predict the digit
        prediction = model.predict([img_array.flatten()])[0]

        print("Your Image Prediction:", prediction)

        # Display the processed image
        plt.imshow(img_array, cmap="gray", vmin=0, vmax=16)
        plt.title(f"Your Image Prediction: {prediction}")
        plt.show()

except FileNotFoundError:
    print("Image not found! Check whether seven.png is on your Desktop.")

except Exception as e:
    print("Error:", e)

