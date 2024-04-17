import { indigo } from "@mui/material/colors";
import { Typography } from "@mui/material";

export default function Overview() {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        flexDirection: "column",
        paddingTop: "40px",
        marginLeft: "15vw",
        marginRight: "15vw",
        marginBottom: "5vh",
      }}
    >
      <Typography
        variant="h5"
        style={{
          color: indigo[800],
          fontFamily: "helvetica",
          fontWeight: 500,
          textAlign: "left",
          marginBottom: "20px",
        }}
      >
        OVERVIEW
      </Typography>
      <Typography
        style={{
          fontFamily: "helvetica",
          fontWeight: 400,
          fontSize: "20px",
          textAlign: "left",
        }}
      >
        A a low-poly art effect renderer for images using C++ and CUDA on GPUs,
        focusing on achieving high-quality effect while enhancing computational
        efficiency through parallel processing
      </Typography>
    </div>
  );
}
