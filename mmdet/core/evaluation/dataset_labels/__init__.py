from .cityscapes_labels import id2label as cs_labels 
from .cityscapes_labels import trainId2label as cs_trainId2label
from .amodalsynthdrive_labels import id2label as asd_labels
from .amodalsynthdrive_labels import trainId2label as asd_trainId2label
__all__ = ['cs_labels', 'asd_labels', 'cs_trainId2label', 'asd_trainId2label']