class Cropper:
    def __init__(self, mins, maxs):
        self.mins = tuple(mins)
        self.maxs = tuple(maxs)

    def crop(self, im, rescale_factor=1):
        if rescale_factor != 1:
            im = (
                im.repeat(rescale_factor, axis=0)
                .repeat(rescale_factor, axis=1)
                .repeat(rescale_factor, axis=2)
            )

        im = im[
            self.mins[0] : self.maxs[0],
            self.mins[1] : self.maxs[1],
            self.mins[2] : self.maxs[2],
        ]
        return im
