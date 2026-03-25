from PIL import Image
import struct
import ctypes

img = Image.open('in.png')
(w, h) = img.size[0:2]
pix = img.load()
buff = ctypes.create_string_buffer(4 * w * h)
offset = 0
for j in range(h):
    for i in range(w):
        r = bytes((pix[i, j][0],))
        g = bytes((pix[i, j][1],))
        b = bytes((pix[i, j][2],))
        a = bytes((255,))
        struct.pack_into('cccc', buff, offset, r, g, b, a)
        offset += 4
out = open('in.data', 'wb')
out.write(struct.pack('ii', w, h))
out.write(buff.raw)
out.close()