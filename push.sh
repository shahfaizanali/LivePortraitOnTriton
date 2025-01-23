#!/bin/bash

# Access the entire string argument as $1
COMMIT_MESSAGE="$1"

# Use it in a git command
git add .
git commit -m "$COMMIT_MESSAGE"
git push origin master -f

