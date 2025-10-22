--[[
  Companion Failsafe Handler Script
  
  This script runs on the flight controller (e.g., ArduPilot) and monitors
  a heartbeat signal from a Python script running on the companion computer.

  If the heartbeat signal is not updated within a specified timeout, the
  script performs failsafe actions:
  1. Attempts to restart the companion script (via custom command).
  2. Commands the vehicle into RTL (Return to Launch) mode.

  --------------------------------------------------------------------------
  REQUIRED COMPANION PYTHON LOGIC:
  The Python script must periodically call a MAVLink library function (like
  pymavlink's set_param_float) to continuously update a temporary parameter
  on the flight controller, for example: 'SCR_USER_VAR_1'.
  
  Example in Python (Pseudo-code):
    import time
    while True:
      # Set the variable to current system time or a simple incrementing counter
      mavlink_connection.set_param('SCR_USER_VAR_1', time.time())
      time.sleep(0.5)
  --------------------------------------------------------------------------
]]

-- CONFIGURATION
local HEARTBEAT_PARAM_NAME = "SCR_USER1" -- Parameter to monitor for the heartbeat signal (Python must update this!)
local TIMEOUT_SECONDS = 3.0                     -- Time in seconds before triggering failsafe
local RTL_MODE_NUMBER = 6                       -- ArduPilot mode number for RTL
local RESTART_ATTEMPTS = 3                      -- Number of times to try restarting before forcing RTL

-- STATE
local last_heartbeat_time = 0                   -- Stores the last time the heartbeat was updated
local failsafe_triggered = false                -- Flag to ensure failsafe actions only run once
local restart_count = 0                         -- Tracks how many restart attempts have been made

-- MAVLink Command Definitions (These are placeholders)
-- You would need to configure your companion computer's MAVLink service
-- (e.g., MAVProxy) to listen for a specific MAVLink command and execute a
-- local shell script (like `supervisorctl restart my_python_script`).
local MAV_CMD_RESTART_SCRIPT = 1000             -- Example Custom Command ID (Must be < 10000)

-- Function to send a MAVLink command to the companion computer
local function send_restart_command()
    if restart_count < RESTART_ATTEMPTS then
        -- This attempts to send a command to the companion computer (GCS/Companion)
        -- to execute a local action (e.g., restart the script).
        
        -- We will use COMMAND_LONG as it's general purpose.
        -- param1=1: Arbitrary identifier for the script to restart.
        gcs:send_cmd(MAV_CMD_RESTART_SCRIPT, 1, 0, 0, 0, 0, 0, 0)
        
        gcs:send_text(4, string.format("Companion: Restart attempt #%d sent.", restart_count + 1))
        restart_count = restart_count + 1
    else
        gcs:send_text(4, "Companion: Max restart attempts reached. Proceeding to RTL.")
    end
end


-- Function to execute the primary failsafe actions
local function trigger_failsafe()
    -- Only run the failsafe sequence once
    if failsafe_triggered then return end
    failsafe_triggered = true

    -- 1. Send status text to the GCS
    gcs:send_text(3, "CRITICAL: Companion Failsafe! Heartbeat lost.")

    -- 2. Attempt to restart the companion script
    send_restart_command()
    
    -- 3. Command RTL
    -- We wait a moment to see if the restart command immediately brings the
    -- script back, but generally we should proceed to a safe mode.
    
    -- We can force RTL immediately or after a short delay (e.g., 5 seconds)
    -- to allow the restart command to take effect. For safety, we force it now.
    local current_mode_num = vehicle:get_mode()
    if current_mode_num ~= RTL_MODE_NUMBER then
        local success = vehicle:set_mode(RTL_MODE_NUMBER)
        if success then
            gcs:send_text(3, "Failsafe: Vehicle mode set to RTL.")
        else
            gcs:send_text(1, "Failsafe ERROR: Could not set mode to RTL!")
        end
    end
end


-- The main update loop, called at high frequency by the autopilot
function update()
    -- Get the current time in seconds
    local current_time = os.time()
    
    -- 1. Read the Heartbeat Value (SCR_USER_VAR_1)
    -- This value is assumed to be updated by the Python script with a timestamp.
    local heartbeat_value = param:get_float(HEARTBEAT_PARAM_NAME)
    
    -- In a real setup, the Python script would write the current time.
    -- If the value is 0, it likely means the Python script hasn't run yet or failed immediately.
    if heartbeat_value == 0 then
        -- Handle the initial case: if script has never run, don't trigger failsafe yet.
        -- This simple example will assume that once a mission starts, the script should be running.
        -- If SCR_USER_VAR_1 is updated, we reset the last_heartbeat_time.
        -- The value read from the parameter is used as the last seen time.
        last_heartbeat_time = current_time
        return
    end

    -- 2. Check if the parameter has been updated since the last check
    -- A simple way to check for 'new data' is to check if the value is different
    -- from a value we internally track, but since the Python script is writing
    -- the CURRENT TIME, we can just use the value directly if it's recent.
    
    -- For robustness, we track when we last saw a "recent" heartbeat value.
    -- If the value is within the last 5 seconds, we consider it fresh and reset the failsafe state.
    if current_time - heartbeat_value < 5.0 then
        -- Heartbeat is fresh! Reset failsafe state.
        last_heartbeat_time = current_time
        if failsafe_triggered then
            gcs:send_text(4, "Companion Failsafe CLEARED. Heartbeat restored.")
            failsafe_triggered = false
            restart_count = 0 -- Reset restart counter upon successful recovery
        end
        return
    end

    -- 3. Check for Timeout
    local time_since_last_fresh_heartbeat = current_time - last_heartbeat_time
    
    if time_since_last_fresh_heartbeat > TIMEOUT_SECONDS then
        -- Heartbeat timed out!
        if not failsafe_triggered then
            -- Initial trigger
            gcs:send_text(3, string.format("WARNING: Companion Heartbeat lost for %.1f seconds.", time_since_last_fresh_heartbeat))
            trigger_failsafe()
        elseif restart_count < RESTART_ATTEMPTS and current_time % 5 == 0 then
            -- Resend restart command every 5 seconds until max attempts are reached
            send_restart_command()
        end
        
        -- Force RTL if max restarts were reached and we are not already in RTL
        if restart_count >= RESTART_ATTEMPTS and vehicle:get_mode() ~= RTL_MODE_NUMBER then
            trigger_failsafe()
        end
    end
end

-- Initialization (run once when the script loads)
-- We set the initial last seen time to ensure we don't trigger the failsafe immediately.
last_heartbeat_time = os.time()
gcs:send_text(6, "Companion Failsafe Handler loaded. Monitoring heartbeat on SCR_USER_VAR_1.")
gcs:send_text(6, "Timeout set to " .. TIMEOUT_SECONDS .. "s.")

return update
