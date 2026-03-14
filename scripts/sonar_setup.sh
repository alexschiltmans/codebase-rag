#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TOKEN_FILE="$PROJECT_DIR/.sonar-token"

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

# Change default password
SONAR_PASS="***REMOVED***"
curl -s -u "$ADMIN_USER:$INITIAL_PASS" -X POST "http://localhost:9000/api/users/change_password" \
  -d "login=admin&previousPassword=admin&password=$SONAR_PASS" > /dev/null 2>&1 || true

# Create project
curl -s -u "admin:$SONAR_PASS" -X POST "http://localhost:9000/api/projects/create" \
  -d "name=codebase-rag&project=codebase-rag" > /dev/null

# Generate token
TOKEN_RESPONSE=$(curl -s -u "admin:$SONAR_PASS" -X POST "http://localhost:9000/api/user_tokens/generate" \
  -d "name=scan-token")
TOKEN=$(echo "$TOKEN_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])" 2>/dev/null || echo "FAILED_TO_EXTRACT")

# Save token for reuse
echo "$TOKEN" > "$TOKEN_FILE"

echo ""
echo "=== Setup Complete ==="
echo "SonarQube URL: http://localhost:9000"
echo "Login:         admin / $SONAR_PASS"
echo "Token saved to .sonar-token"
echo ""
echo "Run the scan:"
echo "  make sonar-scan"
