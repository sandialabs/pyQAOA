import abc

class DocumentationInheritance(type):
    """
    Provides automatic inheritance of docstrings for methods not overridden by derived classes
    """

    def __new__(metaclass, classname, bases, classdict):
        cls = super().__new__(metaclass, classname, bases, classdict)
        for k,v in classdict.items():
            if not getattr(v,'__doc__') and v in dir(super()):
                v.__doc__ = getattr(bases[-1],k).__doc__
        return cls
