import subprocess

# Intern profile
intern_profile = {
    "name": "Zara",
    "skills": ["Python", "Data Analysis"],
    "experience": "Worked on cleaning and visualizing datasets."
}

# Job description
job_description = {
    "title": "Data Science Intern",
    "requirements": ["Python", "Pandas", "Basic ML"],
    "description": "Looking for someone to assist with data tasks and simple ML models."
}

# Prompt for the model
prompt = f"""
You are an AI interview assistant.

Generate 3 technical and 2 behavioral interview questions for a candidate applying as a {job_description['title']}.

Candidate Skills: {", ".join(intern_profile['skills'])}
Experience: {intern_profile['experience']}
Job Requires: {", ".join(job_description['requirements'])}
"""

# Function to call LLaMA via Ollama using --prompt
def call_llama(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3", "--prompt", prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode()

# Call the model
response = call_llama(prompt)

# Print the response
print("\n🎯 Generated Interview Questions:\n")
print(response)
