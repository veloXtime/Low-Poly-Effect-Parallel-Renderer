import React, { useRef } from "react";
import TopBar from "./components/TopBar";
import Overview from "./components/Overview";
import Schedule from "./components/Schedule";
import { Typography, Link } from "@mui/material";
import { grey, indigo } from "@mui/material/colors";
import "./App.css";

function App() {
  const gradient = `linear-gradient(to bottom, ${grey[50]} 10%, ${indigo[100]})`;
  const overviewRef = useRef(null);
  const scheduleRef = useRef(null);

  return (
    <div
      style={{
        paddingTop: "60px", // Adjust top padding to make space for the fixed TopBar
        minHeight: "100vh",
        backgroundImage: gradient,
        backgroundAttachment: "fixed",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <TopBar overviewRef={overviewRef} scheduleRef={scheduleRef} />
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyItems: "center",
          flexDirection: "row",
          justifyContent: "center",
          paddingTop: "40px",
        }}
      >
        <Typography
          variant="h3"
          style={{
            fontFamily: "Roboto, sans-serif",
            fontWeight: 700,
            color: indigo[800],
          }}
        >
          Low-poly Parallel Renderer
        </Typography>
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyItems: "center",
          flexDirection: "row",
          justifyContent: "center",
          paddingTop: "20px",
        }}
      >
        <Typography
          variant="h5"
          style={{
            fontFamily: "Roboto, sans-serif",
            fontWeight: 400,
            marginRight: "20px",
          }}
        >
          <Link
            href="https://github.com/veloXtime"
            underline="none"
            color={indigo[300]}
          >
            Felicity Xu ,
          </Link>
        </Typography>
        <Typography
          variant="h5"
          style={{
            fontFamily: "Roboto, sans-serif",
            fontWeight: 400,
          }}
        >
          <Link
            href="https://github.com/aow-otto"
            underline="none"
            color={indigo[300]}
          >
            Ao Wang
          </Link>
        </Typography>
      </div>
      <div ref={overviewRef}>
        <Overview />
      </div>
      <div ref={scheduleRef}>
        <Schedule />
      </div>
    </div>
  );
}

export default App;
