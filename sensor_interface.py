import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import airsim
from std_msgs.msg import Float32
import os
import numpy as np
import time
import open3d as o3d
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import multiprocessing
import sys
from collections import deque 
from vanderbilt_utils.interface_definitions import message_settings
import vanderbilt_interfaces.msg
import msgpackrpc 
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

class ReturnValueThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = None

    def run(self):
        if self._target is None:
            return  
        try:
            self.result = self._target(*self._args, **self._kwargs)
        except Exception as exc:
            print(f'{type(exc).__name__}: {exc}', file=sys.stderr)  

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self.result

class SensorInterface(Node):
    def __init__(self):
        super().__init__('SensorInterfaceNode',
                allow_undeclared_parameters=True,
                automatically_declare_parameters_from_overrides=True)
        self.cam = self.get_parameter_or(
            'cam', Parameter('cam', Parameter.Type.INTEGER, 0))  
        self.cam = self.cam.value
        
        self.camera_response_list = []        
        self.camera_messages = []
        
        self.vehicle_name = self.get_parameter_or(
            'vehicle_name', Parameter('vehicle_name', Parameter.Type.STRING, "SimpleFlight"))  
        self.vehicle_name = self.vehicle_name.value
        
        self.host_ip = self.get_parameter_or(
            'host_ip', Parameter('host_ip', Parameter.Type.STRING, "127.0.0.1"))  #
        self.host_ip = self.host_ip.value
        
        print("host ip: %s" % self.host_ip)                     
        
        self.camera_publisher = self.create_publisher(
            msg_type=vanderbilt_interfaces.msg.CameraArray,
            topic="/adk_node/vanderbilt/camera_array",
            qos_profile=rclpy.qos.QoSProfile(
                history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE)    
        )
        
        timer1_cb_group = ReentrantCallbackGroup()
        timer3_cb_group = ReentrantCallbackGroup()
        
        self.camera_update= 1/1
        self.create_timer(self.camera_update, self.camera_thread, callback_group=timer1_cb_group)
        
        
        self.sensor_output_update= 1
        self.create_timer(self.sensor_output_update, self.sensor_output_thread, callback_group=timer3_cb_group)
        
    def sensor_output_thread(self):
        _t = ReturnValueThread(target=self.sensor_output, args=())
        _t.start()
        time.sleep(0.1) #s
        _t.join()
                
    def sensor_output(self):
        camera_array_msg = vanderbilt_interfaces.msg.CameraArray()
        camera_array_msg.cameras = []
        if len(self.camera_response_list) > 0:
            lock = threading.Lock()        
            with lock:
                for responses in self.camera_response_list:
                    for r in responses:
                        camera_msg = vanderbilt_interfaces.msg.Camera()
                        # image_update= (r[1].time_stamp - r[0].time_stamp)/1e6   
                        
                        # Get odomerty from image response
                        diff = r[1].time_stamp - r[0].time_stamp #ns (usually 0 or 3 ms)
                        ts = r[0].time_stamp/1e6
                        camera_msg.odomerty.header.stamp.sec = int(ts/1e9)
                        camera_msg.odomerty.header.stamp.nanosec = int(ts-camera_msg.odomerty.header.stamp.sec*1e9)
                        
                        camera_msg.odomerty.pose.pose.position.x = r[0].camera_position.x_val
                        camera_msg.odomerty.pose.pose.position.y = r[0].camera_position.y_val
                        camera_msg.odomerty.pose.pose.position.z = r[0].camera_position.z_val
                        camera_msg.odomerty.pose.pose.orientation.x = r[0].camera_orientation.x_val
                        camera_msg.odomerty.pose.pose.orientation.y = r[0].camera_orientation.y_val
                        camera_msg.odomerty.pose.pose.orientation.z = r[0].camera_orientation.z_val
                        camera_msg.odomerty.pose.pose.orientation.w = r[0].camera_orientation.w_val
                        print(camera_msg.odomerty.pose.pose)
                       
                        camera_msg.segmentation.data = r[0].image_data_uint8
                        camera_msg.segmentation.header = camera_msg.odomerty.header
                        camera_msg.segmentation.height = r[0].height
                        camera_msg.segmentation.width = r[0].width
                        
                        depth = np.resize(r[1].image_data_float, np.size(r[1].image_data_float) * 4 ).astype(np.uint8).tolist()
                        print(np.shape(r[1].image_data_float))
                        camera_msg.depth.data = depth  
                        camera_msg.depth.header = camera_msg.odomerty.header
                        camera_msg.depth.height = r[1].height
                        camera_msg.depth.width = r[1].width
                        camera_msg.depth.encoding = "32FC1"
                         
                        camera_msg.difference = Float32(data=float(diff))
                        
                        camera_array_msg.cameras.append(camera_msg)
            
            self.camera_publisher.publish(camera_array_msg)
            self.get_logger().info(str(len(camera_array_msg.cameras)))
            sys.stdout.flush()       
            self.camera_response_list = []
                  
    # def proc(self):
    #     _p = multiprocessing.Process(target=self.image_callback, args=(self.cam,))
    #     _p.start()
    
    def camera_thread(self):
        _t = ReturnValueThread(target=self.image_callback, args=(self.cam,))
        _t.start()
        time.sleep(self.camera_update) #s
        result = _t.join()
        self.camera_response_list.append(result)
   
        
    def image_callback(self, cam):
        client = airsim.MultirotorClient(self.host_ip)
        responses = []
        request_ns = time.time_ns()
        responses.append(client.simGetImages([
            # Scene uint8 compressed
            # airsim.ImageRequest(cam, 0, False, False), 
            # Segmentation uint8 compressed
            airsim.ImageRequest(cam, 5, False, False),
            # Depth floating point uncompressed)
            airsim.ImageRequest(cam, 1, True, False)]))   
        
        if responses is not None:
            return responses
        else:
            return None     
    
def main(args=None):
    rclpy.init(args=args)
    node = SensorInterface()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()