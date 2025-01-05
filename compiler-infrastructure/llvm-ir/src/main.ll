; Module definition
define i32 @add(i32 %a, i32 %b) {
entry:
    ; Add the two parameters
    %result = add i32 %a, %b

    ; Return the result
    ret i32 %result
}

; Function that uses the add function
define i32 @main() {
entry:
    ; Allocate space for variables
    %x = alloca i32
    %y = alloca i32

    ; Store initial values
    store i32 10, i32* %x
    store i32 32, i32* %y

    ; Load values from memory
    %x_val = load i32, i32* %x
    %y_val = load i32, i32* %y

    ; Call the add function
    %result = call i32 @add(i32 %x_val, i32 %y_val)

    ; Return the result
    ret i32 %result
}
