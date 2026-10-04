# Additional real-image positive control

Saved before selecting or scoring the control image. This checks the actual
candidate gate and full-resolution verification on a known geometric relationship
within one supplied image. It does not add any organiser source-ID truth.

From the valid audit's BSE file manifest, choose the image with the lexically
smallest file SHA-256, without class labels, predictions or visual screening.
For all three matched detector channels of that site, create two 800 x 2,200 px
crops with source origins `(y=160, x=500)` and `(y=253, x=733)`.
The second crop undergoes fixed gain 0.65, offset 31 and nearest-integer uint8
quantisation. These operations cannot clip an original uint8 pixel.
The expected A-minus-B coordinate translation is `(dy=93, dx=233)`.

Use the final audit's unmodified blur/lattice, minimum overlap, candidate gate,
fine registration, three separated patches and three-channel thresholds. Success
requires passing the **actual coarse candidate gate**, exact recovery of the
integer translation and all nine verification checks. Include a fixed-seed
pixel-shuffled BSE negative control that must fail the coarse gate; no shuffled
image enters model training or a dataset.

Save chosen source hashes, source/derived crop coordinates, the scalar checks and
an explicitly labelled positive-control BSE panel. The output is a supplementary
technical control; it is not an overlap between two organisers' supplied sites.
No detector labels, classifier family, thresholds or grouping rule changes here.
