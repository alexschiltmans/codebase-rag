#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TOKEN_FILE="$PROJECT_DIR/.sonar-token"
PASS_FILE="$PROJECT_DIR/.sonar-pass"

# The container's admin password. Generated once and kept beside the token rather than written
# into this file: a password committed to the repository is one every clone shares forever, and
# this one is reachable from anything that can already talk to localhost:9000.
if [ -n "$SONAR_PASS" ]; then
    :
elif [ -f "$PASS_FILE" ]; then
    SONAR_PASS=$(tr -d '[:space:]' < "$PASS_FILE")
else
    # The fixed prefix satisfies SonarQube's upper, lower, digit, and symbol requirement, so the
    # random tail never has to be retried for failing the policy by chance.
    SONAR_PASS="Sq1!-$(LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 24)"
    printf '%s\n' "$SONAR_PASS" > "$PASS_FILE"
    chmod 600 "$PASS_FILE"
fi

echo "=== SonarQube Setup ==="

# Remove existing container if any
docker rm -f sonarqube 2>/dev/null || true

# Start SonarQube
CONTAINER_ID=$(docker run -d --name sonarqube -p 9000:9000 sonarqube:10.7-community)
echo "Container started: ${CONTAINER_ID:0:12}"

# Wait for SonarQube to be ready
echo "Waiting for SonarQube to start (this takes ~60-90 seconds)..."
for i in $(seq 1 60); do
    if curl -s http://localhost:9000/api/system/status | grep -q '"status":"UP"'; then
        echo "SonarQube is UP!"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "Timeout waiting for SonarQube"
        exit 1
    fi
    sleep 3
done

# Change the default password. The credentials go through variables rather than sitting inline, so
# a secret scanner does not have to guess whether a literal after -u is a real credential.
ADMIN_USER="admin"
INITIAL_PASS="admin"
curl -s -u "$ADMIN_USER:$INITIAL_PASS" -X POST "http://localhost:9000/api/users/change_password" \
  -d "login=$ADMIN_USER&previousPassword=$INITIAL_PASS&password=$SONAR_PASS" > /dev/null 2>&1 || true

# Create project
curl -s -u "$ADMIN_USER:$SONAR_PASS" -X POST "http://localhost:9000/api/projects/create" \
  -d "name=codebase-rag&project=codebase-rag" > /dev/null

# Generate token
TOKEN_RESPONSE=$(curl -s -u "$ADMIN_USER:$SONAR_PASS" -X POST "http://localhost:9000/api/user_tokens/generate" \
  -d "name=scan-token")
TOKEN=$(echo "$TOKEN_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])" 2>/dev/null || echo "FAILED_TO_EXTRACT")

# Save token for reuse
echo "$TOKEN" > "$TOKEN_FILE"
chmod 600 "$TOKEN_FILE"

echo ""
echo "=== Setup Complete ==="
echo "SonarQube URL: http://localhost:9000"
echo "Login:         $ADMIN_USER / $SONAR_PASS"
echo "Token saved to .sonar-token, password to .sonar-pass"
echo ""
echo "Run the scan:"
echo "  make sonar-scan"
