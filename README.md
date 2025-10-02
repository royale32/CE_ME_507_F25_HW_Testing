## Purpose

The purpose of this GitHub repository is to enable you to test your finite element analysis software prior to submitting it for credit.

## Disclaimers

Because there is no requirement that you submit your completed homework here to be evaluated and scored, there also should be no expectation that this testing framework is 100% accurate.
Effort has been made to help with software validation, but it is your responsibility to ensure that your code functions regardless of whether or not these tests are passed.
Particularly, it would be very easy to create functions that are designed to pass all of the tests (each of which can be read by you) but that does not actually do any finite element analysis.
The intention of this repository is to aid in your learning efforts, and this repository is a service to you as a student that I am providing.

Finally, this repository will only be helpful in creating and evaluating code that you develop.
It will not help with writen homework, with homework where you are asked to comment code, or on assignments where you are asked to create and/or interpret plots with code.
Please plan accordingly.

## Structure of the Repository

The GitHub repository is organized by assignment. 
Incomplete versions of each assignment are provided in each of the folders, which are identified both by the order of the homework assignment and by the material covered in that homework.
Tests to help you determine if the software that you created is correct or not are also present in each of these folders.

In addition to the homework folder, there is a .github/ folder present in this repository.
This folder is where I tell GitHub how to check run your assignments when you upload changes to your version of the repository.
While this folder will be updated regularly, you should never need to update anything in this folder.

As you are aware, many of the homework assignments that we complete depend on previous homework assignments.
This interdependency can be accomplished in two ways:
  - Copy and paste a lot of code between different assignments to allow your latest assignment to use all of your previous code, or
  - Use referencing to import other assignments and call functionality developed therein.
While the first of these is allowable in this testing framework and may be more convenient at first, it will create headaches because you will have multiple different versions of a single file.
If you need to make changes later on, those changes will not appear in all of your previous versions and you will have a big mess on your hands.
Alternatively, the second of these methods requires a bit more housekeeping, but is far cleaner and more sustainable.
Instructions will be given so that you can be successful in the latter of these two approaches.

## Testing Your Code

It is assumed that you have little to no experience using GitHub.
If this is the case, these instructions are for you.
In any case, I will be describing the absolute simplest way that I can for navigating this testing framework.
It will not involve terminal or command prompt, though if you know how to use these in conjunction with GitHub you should be able to read between the lines to operate using these methods.

To test your code, please log into your GitHub account (perhaps after creating one first), navigate to this repository, and fork the repository.
Instructions on how to do this for a general repository are available online, and you can search for those on your own.
While you can create another repository from this one as a template, it is highly recommended that you do not because then you will have no hopes of updating your repository as I continue to add new testing functionality to this framework.
That discussion will be postponed until later.

We will begin with the univariate Lagrange polynomials homework, which is not dependent on code from any other homework.
To test your homework, please rename your homework assignment to have *exactly* the same name as the (incomplete) file in this folder (HW3).
Furthermore, all of your functions should have *exactly* the same names and same input---otherwise the software may fail.
After you have renamed your file, please run it on your own machine to make sure that it still works.
When you are done, click on the Homework 3 folder in your GitHub fork of this repository, select "Add File -> Upload Files" in the upper right area of your screen, and upload your homework 3 code.

After uploading your file (to override the current file), you can click on the "Actions" tab in GitHub, near the top middle, to see if your code passes all tests associated with Homework 3.
This can be checked by clicking on the topmost workflow run, selecting the "build" label, and clicking on "Test Univariate B-Splines (HW3) with unittest."
Your code will not work because I changed the "InterpolateFunction" command to no longer plot, but instead return the xis and ys variables (in that order).
If you would also like to change your functions and then create a new plotting function that uses this interpoation data, you should be able to have it pass all tests.
Otherwise, feel free to ignore this error and move on.

Next, depending on how you completed Homework 6, you may rely on the code from Homework 3 as a dependency.
This would mean that rather than copying and pasting your homework 3 code into the top of your homework 6 files, you imported the functionality from homework 3 into your homework 6 code using an "import" command.
I will describe this framework.

In this GitHub repository, all homework is separated into different folders for cleanliness.
To call information from Homework 3 in Homework 6, I need homework 3 to be visible to my Homework 6 files.
On your own machine, create folders for Homework 3 and Homework 6 *exactly* as done on the GitHub repository, both in the same parent folder.
Put your Homework 3 and Homework 6 code in their respective folders, named *exactly* as they are on GitHub.
Then, in your Homework 6 files, place the following near the top of your file (next to other import commands).

import sys
sys.path.append('../HW3_Univariate_Lagrange/')
import Univariate_Lagrange_Basis_Functions as <_insert what you'd like to call it here_>

If I called the import HW3, to then call the LagrangeBasisEvaluation from Homework 3, I would write in my HW6 file "HW3.LagrangeBasisEvaluation(<inputs here>)"
Consequently, I would be referencing Homework 3 files but not copying/pasting or replicating their functionality.
Any updates to HW3 files are immediately seen by all files that rely on HW3.

Using this dependency information on my own machine, I then test and make sure that my software still functions the way I expect.
If so, you can then open your forked GitHub repository, upload your Homework 6 files into the Homework 6 folder (with files named *exactly* the same as they are on the GitHub repository), and then check the actions.
If you are successful, you will see that your Homework 6 tests now run correctly.
Passing tests that correctly work in this framework are marked with a check mark; tests that fail are marked with an X.

Proceed in like manner for all other homeworks as done for homework 6.
When finished, you likely have a cleaner coding framework that reduces redundancy and you also should get a green check mark on your testing framework that indicates that your software runs correctly.

## Updating Your Fork

When you initially forked your repository, it had testing for only some of the homework.
However, there are more coding assignments in the class that were not listed, but whose tests are still being developed.
To access these, you will either need to create a new repository and repeat all of the above steps again or you will need to learn how to update a forked repository.

Instructions are not given now for updating a forked repository based on edits made in the original repository, but they will be made hereafter. 
Please check for additional information at a later date.

## Contributing

This is a work in progress, that aims to be a service to the interested student.
If you are using this and have any challenges, please let me know and I would be happy to try to understand your challenges and update this documentation so that things are easier to use.
Thank you for your feedback, and thank you for being patient as this becomes more functional.


