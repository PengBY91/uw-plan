from pathlib import Path
import hashlib,json
from rosbags.highlevel import AnyReader
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu,Image
p=Path('/home/steve/uw-ao-lab/data/public/euroc/V1_01_easy_ros2')
out={}
with AnyReader([p]) as r:
 for c,t,raw in r.messages():
  if c.topic not in ['/imu0','/cam0/image_raw','/cam1/image_raw'] or c.topic in out:continue
  a=r.deserialize(raw,c.msgtype);b=deserialize_message(bytes(raw),Imu if c.topic=='/imu0' else Image)
  if c.topic=='/imu0':
   av=[getattr(getattr(a,g),k) for g in ['angular_velocity','linear_acceleration'] for k in ['x','y','z']]
   bv=[getattr(getattr(b,g),k) for g in ['angular_velocity','linear_acceleration'] for k in ['x','y','z']]
   assert av==bv;out[c.topic]={'equal':True,'values':bv}
  else:
   assert bytes(a.data)==bytes(b.data);out[c.topic]={'equal':True,'pixel_sha256':hashlib.sha256(bytes(b.data)).hexdigest()}
  if len(out)==3:break
print(json.dumps(out,indent=2))
