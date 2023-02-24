from threading import Event

import rospy
import smach

from aruco_opencv_msgs.msg import ArucoDetection


class StartState(smach.State):
    """State checking if the board is seen by the rover"""

    def __init__(
        self,
        outcomes=["board_not_found", "board_found", "preempted"],
        input_keys=["action_goal", "action_feedback", "action_result"],
        timeout=3.0,
        name="Start",
    ):
        super().__init__(outcomes, input_keys)
        if rospy.has_param("~start_state/timeout"):
            self.timeout = rospy.get_param("~start_state/timeout", timeout)
        else:
            self.timeout = rospy.get_param("~timeout", timeout)

        self.board_flag: Event = Event()

        self.state_log_name = name

        self.reset_state()

    def reset_state(self):
        self.board_id = None
        self.board_flag.clear()

    def board_callback(self, data: ArucoDetection):
        """Function called every time there is new ArucoDetection message published on the topic.
        Checks if the rover can see board with the desired id (passed as action goal).
        """
        # if board is not seen yet and there are any boards detected
        if len(data.boards) != 0 and not self.board_flag.is_set():
            # look for the desired board
            for board in data.boards:
                if board.board_name == self.board_id:
                    self.board_flag.set()
                    break

    def service_preempt(self):
        """Function called when the state catches preemption request.
        Removes all the publishers and subscribers of the state.
        """
        rospy.logwarn(f"Preemption request handling for '{self.state_log_name}' state.")
        self.board_sub.unregister()
        return super().service_preempt()

    def execute(self, ud):
        """Main state method, executed automatically on state entered"""
        self.reset_state()

        self.board_id = ud.action_goal.board_id
        rospy.loginfo(
            f"Waiting for board detection. Required board_id: {self.board_id}"
        )

        self.board_sub = rospy.Subscriber(
            "aruco_detections", ArucoDetection, self.board_callback, queue_size=1
        )

        rate = rospy.Rate(10)
        time_start = rospy.Time.now()
        while not self.board_flag.is_set():
            if self.preempt_requested():
                self.service_preempt()
                ud.action_result.result = f"{self.state_log_name}: state preempted."
                return "preempted"

            if (rospy.Time.now() - time_start).to_sec() > self.timeout:
                self.board_sub.unregister()
                rospy.logerr("Didn't find a board. Docking failed.")
                ud.action_result.result = (
                    f"{self.state_log_name}: didn't find a board. Docking failed."
                )
                return "board_not_found"

            rate.sleep()

        self.board_sub.unregister()

        ud.action_feedback.current_state = (
            f"{self.state_log_name}: board with id: {self.board_id} found. "
            f"Proceeding to 'Check Area' state."
        )
        return "board_found"
