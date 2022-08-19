# Image Transformations

## Geometric Transformations

- Types of Geometric Transformations:
  - Geometric
  - Pixel coordinate
  - Brightness interpolation

- Applications
  - Computer graphics
  - Distortion - Introduce / Eliminate
  - Image processing / Preprocessing
  - Text recognition
  - Recognition of signs, numbers, etc

## Formulation of the problem

- Distorted image $f(i, j)$

- Corrected image $f'(i', j')$

- Mapping between the images
  $$
  i = T_i(i', j')\quad j = T_j(i', j')
  $$

- Define the Transformation
  - Known in advance / determine through correspondences
  - Image to know
  - Image to image

- Apply the defined Transformation
  - For every point in the output image
  - Determine where it came from using T from mapping
  - Interpolate a value for the output point

## Transformations

- Linear Transformation in 3D
  - Affine Transformation
    - Euclidean ( 6 DoF ) ---- 3 for translation, 3 for rotation
- Higher Order Transformations
  - Parameterized
    - B-splines
  - Freedom
    - Warp field

### Affine Transformations

- Known Transformations
  - Translation, Rotation

- Unknown Transformations
  - Would require at least 3 observations
  - Could be points in an image
    - Corners of objects