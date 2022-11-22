from threading import Event

import rospy
import smach

from aruco_opencv_msgs.msg import MarkerDetection


class StartState(smach.State):
    """State checking if the marker is seen by the rover"""

    def __init__(self, outcomes=["marker_not_found", "marker_found"], timeout=3.0):
        super().__init__(outcomes)
        self.marker_sub = rospy.Subscriber(
            "marker_detections", MarkerDetection, self.marker_callback, queue_size=1
        )
        self.timeout = timeout
        self.marker_flag: Event = Event()

    def marker_callback(self, data: MarkerDetection):
        if len(data.markers) != 0 and not self.marker_flag.is_set():
            self.marker_flag.set()

    def execute(self, userdata):
        self.marker_flag.clear()
        rospy.loginfo("Waiting for marker detection.")

        if not self.marker_flag.wait(self.timeout):
            self.marker_sub.unregister()
            rospy.logerr("Didn't find a marker. Docking failed.")
            return "marker_not_found"

        self.marker_sub.unregister()

        return "marker_found"
