import site, os
for sp in site.getsitepackages():
    f = os.path.join(sp, 'easyocr', 'easyocr.py')
    if os.path.exists(f):
        c = open(f).read()
        if 'from bidi import get_display' in c:
            open(f,'w').write(c.replace('from bidi import get_display','from bidi.algorithm import get_display'))
            print('Patched easyocr bidi import')
        break
