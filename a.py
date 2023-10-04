import numpy as np
import cv2

def image_to_matrix(image):
  """Converts an image to a NumPy matrix.

  Args:
    image: A NumPy array representing the image.

  Returns:
    A NumPy matrix representing the image.
  """

  if len(image.shape) == 2:
    # Grayscale image
    return np.array(image)
  else:
    # Color image
    return np.array(image)[:, :, ::-1]

def matrix_to_image(matrix):
  """Converts a NumPy matrix to an image.

  Args:
    matrix: A NumPy matrix representing the image.

  Returns:
    A NumPy array representing the image.
  """

  if len(matrix.shape) == 2:
    # Grayscale image
    return matrix
  else:
    # Color image
    return matrix[:, :, ::-1]

def save_matrix_to_text_file(matrix, filename):
  """Saves a NumPy matrix to a text file.

  Args:
    matrix: A NumPy matrix to be saved.
    filename: The name of the text file to save the matrix to.
  """

  with open(filename, "w") as f:
    for row in matrix:
      f.write(" ".join([str(element) for element in row]))
      f.write("\n")

# Read the image
image = cv2.imread("image.jpg")

# Convert the image to a matrix
image_matrix = image_to_matrix(image)

# Reverse the process
image_reconstructed = matrix_to_image(image_matrix)

# Save the reconstructed image
cv2.imwrite("reconstructed_image.jpg", image_reconstructed)

# Save the matrix to a text file
save_matrix_to_text_file(image_matrix, "matrix.txt")
