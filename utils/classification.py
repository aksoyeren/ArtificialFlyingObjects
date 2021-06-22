class ClassificationComplexity():
    
    def __init__(self,fineGrained=False):
        self.fineGrained = fineGrained
    
    @property
    def fineGrained(self):
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
        ] if self._fineGrained else [
            'square',
            'triangle',
            'circular',
            'background'
        ]
    @fineGrained.setter
    def fineGrained(self, fineGrained):
        self._fineGrained = fineGrained
        
        return self.fineGrained
    
    def isfineGrained(self):
        return self._fineGrained
        
    def __str__(self):
        return self.fineGrained
    
    def __len__(self):
        return len(self.fineGrained)
        