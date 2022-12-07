from threading import Event

import rospy
import smach

from aruco_opencv_msgs.msg import MarkerDetection


class StartState(smach.State):
    """State checking if the marker is seen by the rover"""

    def __init__(
        self,
        outcomes=["marker_not_found", "marker_found", "preempted"],
        input_keys=["action_goal"],
        timeout=3.0,
        name="Start",
    ):
        super().__init__(outcomes, input_keys)
        if rospy.has_param("~start_state/timeout"):
            self.timeout = rospy.get_param("~start_state/timeout", timeout)
        else:
            self.timeout = rospy.get_param("~timeout", timeout)

        self.marker_flag: Event = Event()
        self.marker_id = None

        self.state_log_name = name

    def marker_callback(self, data: MarkerDetection):
        if len(data.markers) != 0 and not self.marker_flag.is_set():
            for marker in data.markers:
                if marker.marker_id == self.marker_id:
                    self.marker_flag.set()
                break

    def service_preempt(self):
        rospy.logwarn(f"Preemption request handling for '{self.state_log_name}' state.")
        self.marker_sub.unregister()
        return super().service_preempt()

    def execute(self, userdata):
        self.marker_flag.clear()
        self.marker_id = userdata.action_goal.marker_id
        rospy.loginfo(
            f"Waiting for marker detection. Required marker_id: {self.marker_id}"
        )
        self.marker_sub = rospy.Subscriber(
            "marker_detections", MarkerDetection, self.marker_callback, queue_size=1
        )

        if not self.marker_flag.wait(self.timeout):
            self.marker_sub.unregister()
            rospy.logerr("Didn't find a marker. Docking failed.")
            return "marker_not_found"

        if self.preempt_requested():
            self.service_preempt()
            return "preempted"

        self.marker_sub.unregister()

        return "marker_found"
