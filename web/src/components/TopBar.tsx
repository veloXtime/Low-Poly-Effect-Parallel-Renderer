import { Button, styled } from "@mui/material";
import "../App.css";
import { grey, indigo } from "@mui/material/colors";
import React from "react";

const StyledButton = styled(Button)({
  width: "100px",
  marginRight: "10px",
  color: indigo[800],
});

interface TopBarProps {
  overviewRef: React.RefObject<any>;
  scheduleRef: React.RefObject<any>;
}

const TopBar: React.FC<TopBarProps> = ({ overviewRef, scheduleRef }) => {
  const handleButtonClick = (url: string) => {
    window.open(url, "_blank");
  };

  const scrollToSection = (ref: any) => {
    ref.current.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div
      style={{
        position: "fixed", // Fix position at the top
        top: 0,
        left: 0,
        right: 0,
        height: "60px", // Define a fixed height for the bar
        display: "flex",
        alignItems: "center",
        justifyItems: "center",
        flexDirection: "row",
        justifyContent: "flex-end",
        paddingTop: "10px",
        paddingLeft: "5vw",
        paddingRight: "5vw",
        backgroundColor: grey[50],
        zIndex: 1000, // High z-index to keep it above other content
      }}
    >
      <StyledButton
        disableElevation
        onClick={() => scrollToSection(overviewRef)}
      >
        Overview
      </StyledButton>
      <StyledButton
        disableElevation
        onClick={() => scrollToSection(scheduleRef)}
      >
        Schedule
      </StyledButton>
      <StyledButton
        disableElevation
        onClick={() =>
          handleButtonClick(
            process.env.PUBLIC_URL + "/15618_Project_Proposal.pdf"
          )
        }
      >
        Milestone
      </StyledButton>
      <StyledButton
        disableElevation
        onClick={() =>
          handleButtonClick(
            process.env.PUBLIC_URL + "/15618_Milestone_Report.pdf"
          )
        }
      >
        Milestone
      </StyledButton>
      <Button
        variant="contained"
        disableElevation
        style={{ width: "100px", backgroundColor: indigo[800] }}
        onClick={() =>
          handleButtonClick(
            "https://github.com/veloXtime/Low-Poly-Effect-Parallel-Renderer"
          )
        }
      >
        Github
      </Button>
    </div>
  );
};
export default TopBar;
