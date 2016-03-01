import re
TYPE_RE = re.compile(r'(?:int|char)(?:\s+*?\s|\s*?\s+)([a-zA-Z_].*)')


def setup(app):
    app.connect('autodoc-process-signature', process_signature)


def process_signature(app, what, name, obj, options, signature, return_annotation):
    # Some unused arguments
    # pylint: disable=W0613

    if what not in ('function', 'method'):
        return

    new_params = list()
    for param in (x.strip() for x in signature[1:-1].split(',')):
        hit = TYPE_RE.match(param)
        if hit:
            new_params.append(hit.group(1))
        else:
            new_params.append(param)

    return ('(%s)' % ', '.join(new_params), return_annotation)
