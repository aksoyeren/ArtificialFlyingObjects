from collections import OrderedDict
class Classes():
    """Define which labels are used for the labs. fineGrained decide if few or many labels should be used."""
    def __init__(self,fineGrained=False):
        self.fineGrained = fineGrained
    
    @property
    def classes(self):
        """ """
        return [
            'square_red',
            'square_green',
            'square_blue',
            'square_yellow',
            'triangle_red',
            'triangle_green',
            'triangle_blue',
            'triangle_yellow',
            'circular_red',
            'circular_green',
            'circular_blue',
            'circular_yellow'
        ] if self.fineGrained else [
            'square',
            'triangle',
            'circular',
            'background'
        ]
    @property
    def fineGrained(self) -> bool:
        """Is fineGrained enabled"""
        return self.__fineGrained
    
    @fineGrained.setter
    def fineGrained(self, fineGrained:bool) -> None:
        """Set fineGrained

        :param fineGrained: bool:
        """
        self.__fineGrained = fineGrained

    def __str__(self):
        return self.classes
    
    def __len__(self):
        return len(self.classes)
    
    @property
    def colormap(self) -> "OrderedDict":
        """Colormap for labels and images"""
        return OrderedDict([
            ("square_red",(128, 0, 0)),
            ("square_green",(0, 128, 0)),
            ("square_blue",(128, 128, 0)),
            ("square_yellow",(0, 0, 128)),
            ("triangle_red",(128, 0, 128)),
            ("triangle_green",(0, 128, 128)),
            ("triangle_blue",(128, 128, 128)),
            ("triangle_yellow",(64, 0, 0)),
            ("circular_red",(192, 0, 0)),
            ("circular_green",(64, 128, 0)),
            ("circular_blue",(192, 128, 0)),
            ("circular_yellow",(64, 0, 128)),
            ("background",(0, 0, 0))
        ]) if self.fineGrained else OrderedDict([
            ("square",(0, 0, 128)),
            ("triangle",(0, 128, 0)),
            ("circular",(128, 0, 0)),
            ("background",(0, 0, 0)),
        ])
  