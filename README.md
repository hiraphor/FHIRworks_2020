# Deployment Guide

## Install Visual Studio Code

- Download and install [Visual Studio Code](https://code.visualstudio.com/)

## Install .NET Core 2.1

- Download and install .NET Core 2.1 [SDK 2.1.803](https://dotnet.microsoft.com/download/dotnet-core/2.1)
- Test your installation by opening a new terminal and running the following command:

    ```bash
    dotnet
    ```

# Running the app

- Clone the project and open it in Visual Studio Code.
- Open the file **appsettings.json**


    ```json
    {
      "Logging": {
        "LogLevel": {
          "Default": "Debug",
          "System": "Information",
          "Microsoft": "Information"
        }
      },
      "Instance": "",
      "Tenant": "",
      "ClientId": "",
      "ClientSecret": "",
      "BaseAddress": "",
      "Scope": ""
    }
    ```
- Replace the empty fields with the Azure FHIR API credentials you have been given.
- Save the file.
- Navigate to the directory **dotnet-azure-fhir-web-api** using the terminal inside Visual Studio Code.
- In the terminal, run the following command:

    ```bash
    dotnet run
    ```
- Open a web browser and navigate to [https://localhost:5001/api/Patient/](https://localhost:5001/api/Patient/) to view a list of all patients.


## List of API endpoints

#### Patients

- GET all patients: **/api/Patient**
- GET a patient: **/api/Patient/** *patient ID*
- GET a selected number of pages of patient: **api/pages/** *number of pages*


#### Observations

- GET all observations for a patient: **/api/Observation/** *patient ID*
- GET a single observation for a patient: **api/Observation/single/** *observation ID*
- GET a selected number of pages of observations for a patient: **api/Observation/pages/** *number of pages/patient ID*


# Adding more controllers

The app uses Dependency Injection (DI) design pattern. For more information on how to implement that on .NET Core check [this article](https://docs.microsoft.com/en-us/aspnet/core/fundamentals/dependency-injection?view=aspnetcore-2.1)


# Logging and debugging

Check [this article](https://code-maze.com/net-core-web-development-part3/) for an overview of the implemented logging features.


# Tool Suite

- A platform that provides access and documentation to FHIR developer tools
- Access the platform at: https://toolsuite.azurewebsites.net/

1) Login with your UCL email
2) Register the application you are creating for the Hackathon
3) Read documentation for chosen tool to gain access
4) Input chosen tools in application profile



