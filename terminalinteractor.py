import subprocess
import pexpect  # You might need to install this package

# Command to run the game
command = 'snap run zork'

# Start the subprocess
p = pexpect.spawn(command, encoding='utf-8')
p.echo = False
# Interact with the subprocess
while True:
    try:
        p.expect('>')  # Adjust this pattern based on how Zork prompts for input
        print(p.before)  # Print the output from Zork

        user_input = input()
        if user_input.lower() == 'exit':  # Add a way to break the loop
            break

        p.sendline(user_input)  # Send input to Zork
    except (pexpect.EOF, pexpect.TIMEOUT):
        print("Game ended or timed out")
        break

p.close()