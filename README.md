웹서버 실행 cd ~/sw_project_2025/ros2_ws/src/analyze_dashboard_pkg/web_dashboard            명령어 : uvicorn dashboard:app --reload
분석 노드 실행 cd ~/sw_project_2025/ros2_ws      colcon build    . install/setup.bash         명령어 : ros2 launch analyze_dashboard_pkg dashboard_launch.py
웹서버 접속  http://localhost:8000

#object의 각 하위 폴더에 같은 객체에 해당하는 사진을 넣어주면 분석결과를 웹서버에서 확인할 수 있음. 
#꼭 웹서버 실행을 먼저한 후에 분석노드를 실행할것.
