from enum import Enum

# abstract enum
class ExtendedEnum(Enum):
	@classmethod
	def list(cls):
		return list(map(lambda c: c.value, cls))

# Experiment training mode
class Mode(str, ExtendedEnum):
	EXP = 'exp'
	DEV = 'dev'
	
# Track carbon mode
class TrackCarbon(str, ExtendedEnum):
	OFFLINE = 'offline'
	ONLINE = 'online'

# Data configuration constant
class InputType(ExtendedEnum):
	SINGLE = 1
	DUAL = 2
	
class SpecToken(str, ExtendedEnum):
	PAD = '<pad>'
	UNK = '<unk>'
	CLS = '<cls>'

class Normalization(str, ExtendedEnum):
	NONE = None
	STANDARD = 'std'
	LOG_STANDARD = 'log_std'
	SOFTMAX = 'softmax'
	LOG_SOFTMAX = 'log_softmax'

# Model configuration constant
class ContextType(str, ExtendedEnum):
	LSTM='lstm'
	CNN='cnn'
	ATTENTION='attention'
	
# Data choice
class Data(str, ExtendedEnum):
	ESNLI='esnli'
	HATEXPLAIN='hatexplain'
	YELPHAT='yelphat'
	YELPHAT50 = 'yelphat50'
	YELPHAT100 = 'yelphat100'
	YELPHAT200 = 'yelphat200'