#!/bin/bash
HOST=$(hostname)
RANK=${OMPI_COMM_WORLD_RANK:-0}
PREFIX="${HOST}(${RANK}): "

FIFO=$(mktemp -u /tmp/wrap_mpi_fifo_${RANK}.XXXXXX)
mkfifo "$FIFO"

sed -u "s/^/${PREFIX}/" < "$FIFO" &
SED_PID=$!

"$@" > "$FIFO" 2>&1 &
CMD_PID=$!

trap "kill -TERM $CMD_PID 2>/dev/null; rm -f $FIFO" TERM INT HUP

wait $CMD_PID
EXIT_CODE=$?

rm -f "$FIFO"
wait $SED_PID 2>/dev/null

exit $EXIT_CODE
