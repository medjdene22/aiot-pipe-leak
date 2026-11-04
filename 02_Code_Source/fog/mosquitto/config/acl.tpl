# ACL for m2i2a/equipeXX (TEAM_ID is substituted by fog/setup_auth.sh)
# Principle of least privilege:
#   - edge nodes: may WRITE their own telemetry/alert/status, READ fog results and model commands
#   - fog service: read-write on the whole team prefix
#   - admin (mosquitto_sub monitoring): read-only

user edge_XX
topic write m2i2a/equipeXX/edge/#
topic read m2i2a/equipeXX/fog/#
topic read m2i2a/equipeXX/edge/commande/#

user fog_XX
topic readwrite m2i2a/equipeXX/#

user admin_XX
topic read m2i2a/equipeXX/#

# Explicit denial examples (everything else is denied by default):
# user edge_XX
# topic deny m2i2a/equipeXX/edge/node2/#
