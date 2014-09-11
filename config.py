from ConfigParser import SafeConfigParser
import os

parser = SafeConfigParser()
parser.read(os.path.expandvars('$BMI3D/config'))

recording_system = dict(parser.items('recording_sys'))['make']
data_path = dict(parser.items('db_config_default'))['data_path']

db_config_default = dict(parser.items('db_config_default'))

window_start_x = dict(parser.items('graphics'))['window_start_x']
window_start_y = dict(parser.items('graphics'))['window_start_y']
display_start_pos = '%s,%s' % (window_start_x, window_start_y)

reward_system_version = int(dict(parser.items('reward_sys'))['version'])
