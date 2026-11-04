#!/usr/bin/env bash
# Generate the Mosquitto password file from .env values (never commit passwd!).
# Usage: cp .env.example .env  # edit values
#        ./setup_auth.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "ERROR: .env not found. Copy .env.example to .env and fill it in." >&2
  exit 1
fi
set -a; . ./.env; set +a

PASSWD=mosquitto/config/passwd
: > "$PASSWD"

add_user() {
  local u="$1" p="$2"
  mosquitto_passwd -b "$PASSWD" "$u" "$p"
  echo "added user $u"
}

add_user "${EDGE_USER}"   "${EDGE_PASSWORD}"
add_user "${FOG_USER}"    "${FOG_PASSWORD}"
add_user "${ADMIN_USER}"  "${ADMIN_PASSWORD}"

# Rewrite the ACL template with the real team id
sed "s/XX/${TEAM_ID}/g" mosquitto/config/acl.tpl > mosquitto/config/acl

chmod 640 "$PASSWD"
echo "Done. Users: ${EDGE_USER}, ${FOG_USER}, ${ADMIN_USER} (team ${TEAM_ID})"
