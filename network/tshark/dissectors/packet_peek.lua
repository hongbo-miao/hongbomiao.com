-- Helper function to convert a string of bytes to a hex string
local function bytes_to_hex_string(byte_str)
    local hex_str = ""
    for i = 1, #byte_str do
        hex_str = hex_str .. string.format("%02x ", string.byte(byte_str, i))
    end
    return hex_str
end

-- This function is called when the script is loaded
local function init_listener()
    print("Lua script loaded! Will print packet details and first 16 bytes.")

    local tap = Listener.new(nil, "frame")

    function tap.packet(pinfo, tvb, userdata)
        print("--------------------------------------------------")
        print("Packet: #" .. pinfo.number ..
              ", Original Length: " .. pinfo.len .. " bytes" ..
              ", Captured Length: " .. pinfo.caplen .. " bytes")

        -- Get the length of the data available in the TVB
        local tvb_len = tvb:len()
        -- Determine how many bytes to print (up to 16, or fewer if packet is smaller)
        local bytes_to_print = math.min(tvb_len, 16)

        if bytes_to_print > 0 then
            -- Extract the raw bytes from the TVB
            -- 1. Get a TvbRange object for the desired byte range.
            --    TVB offsets are 0-indexed.
            local packet_bytes_tvbrange = tvb:bytes(0, bytes_to_print)
            -- 2. Get the raw string data from the TvbRange.
            local packet_bytes_str = packet_bytes_tvbrange:raw()

            -- Convert to hex and print
            print("First " .. bytes_to_print .. " bytes (hex): " .. bytes_to_hex_string(packet_bytes_str))
        else
            print("No data in TVB to print.")
        end
    end

    function tap.draw()
        print("--------------------------------------------------")
        print("Capture drawing/summary phase. 'hello.lua' signing off.")
    end
end

init_listener()
