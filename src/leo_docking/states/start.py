from threading import Event

import rospy
import smach

from aruco_opencv_msgs.msg import MarkerDetection


class StartState(smach.State):
    """State checking if the marker is seen by the rover"""

    def __init__(
        self,
        outcomes=["marker_not_found", "marker_found", "preempted"],
        input_keys=["action_goal", "action_feedback", "action_result"],
        timeout=3.0,
        name="Start",
    ):
        super().__init__(outcomes, input_keys)
        if rospy.has_param("~start_state/timeout"):
            self.timeout = rospy.get_param("~start_state/timeout", timeout)
        else:
            self.timeout = rospy.get_param("~timeout", timeout)

        self.marker_flag: Event = Event()

        self.state_log_name = name

        self.reset_state()

    def reset_state(self):
        self.marker_id = None
        self.marker_flag.clear()

    def marker_callback(self, data: MarkerDetection):
        """Function called every time there is new MarkerDetection message published on the topic.
        Checks if the rover can see marker with the desired id (passed as action goal).
        """
        # if marker not seen yet and there are any markers detected
        if len(data.markers) != 0 and not self.marker_flag.is_set():
            # look for the desired marker
            for marker in data.markers:
                if marker.marker_id == self.marker_id:
                    self.marker_flag.set()
                break

    def service_preempt(self):
        """Function called when the state catches preemption request.
        Removes all the publishers and subscribers of the state.
        """
        rospy.logwarn(f"Preemption request handling for '{self.state_log_name}' state.")
        self.marker_sub.unregister()
        return super().service_preempt()

    def execute(self, ud):
        """Main state method, executed automatically on state entered"""
        self.reset_state()

        self.marker_id = ud.action_goal.marker_id
        rospy.loginfo(
            f"Waiting for marker detection. Required marker_id: {self.marker_id}"
        )

        self.marker_sub = rospy.Subscriber(
            "marker_detections", MarkerDetection, self.marker_callback, queue_size=1
        )

        rate = rospy.Rate(10)
        time_start = rospy.Time.now()
        while not self.marker_flag.is_set():
            if self.preempt_requested():
                self.service_preempt()
                ud.action_result.result = f"{self.state_log_name}: state preempted."
                return "preempted"

            if (rospy.Time.now() - time_start).to_sec() > self.timeout:
                self.marker_sub.unregister()
                rospy.logerr("Didn't find a marker. Docking failed.")
                ud.action_result.result = (
                    f"{self.state_log_name}: didn't find a marker. Docking failed."
                )
                return "marker_not_found"

            rate.sleep()

        self.marker_sub.unregister()

        ud.action_feedback.current_state = (
            f"{self.state_log_name}: marker with id: {self.marker_id} found. "
            f"Proceeding to 'Check Area' state."
        )
        return "marker_found"
