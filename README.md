# PV-monitor-API

To run the API:
```bash
python PV_monitor/app.py
 ```
 
 A request would look like:
 ```
 http://127.0.0.1:5000/classifier?irradiance=1000&temperature=25&voltage=29&current=7.1&power=20
 ```
 
 Output will look like:
 ```json
 {"status": "SUCCESS", "result": "Panel Degradation"}
 ```
