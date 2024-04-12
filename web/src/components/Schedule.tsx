import { indigo } from "@mui/material/colors";
import { Checkbox, List, ListItem, Typography } from "@mui/material";

interface Task {
  task: string;
  completed: boolean;
}

interface WeekPlan {
  week: string;
  tasks: Task[];
}

function TaskItem(props: { task: Task }) {
  return (
    <ListItem
      sx={{
        padding: "0px 20px 0px 20px",
        margin: "10px 0px 10px 0px",
        alignItems: "flex-start", // to keep checkbox on the top
      }}
    >
      <Checkbox
        edge="start"
        checked={props.task.completed}
        tabIndex={-1}
        disableRipple
        style={{
          color: indigo[400],
          margin: "0px",
          marginRight: "20px",
          padding: "0px",
        }}
      />
      <Typography
        style={{
          fontFamily: "helvetica",
          fontWeight: 400,
          fontSize: "20px",
          textAlign: "left",
          width: "100%",
        }}
      >
        {props.task.task}
      </Typography>
    </ListItem>
  );
}

export default function Schedule() {
  const plan: WeekPlan[] = [
    {
      week: "Week 1: March 25 - March 30",
      tasks: [
        {
          task: "Start the basic structure of the renderer in C++ and CUDA. Finish loading and outputing images, viewing images, etc.blahblah blablah blahblah blablah blahblah blablah blahblah blahblah  blahblah",
          completed: true,
        },
        {
          task: "Research any relevant and existing implementations of low-poly rendering and parallel processing techniques.",
          completed: true,
        },
      ],
    },
    {
      week: "Week 2: March 31 - April 6",
      tasks: [
        {
          task: "For first half of the week, outline and finish Gaussian Blur step for the image rendering.",
          completed: true,
        },
        {
          task: "For second half of the week, start coding on the edge drawing algorithm, aim to finish extraction of anchors and edge routing.",
          completed: true,
        },
      ],
    },
    {
      week: "Week 3: April 7 - April 13",
      tasks: [
        {
          task: "Wrap up edge drawing algorithm, including vertices finding. This is the basic version of our edge drawing step.",
          completed: false,
        },
        {
          task: "Attempt a basic version of delaunay algorithm and test the baseline performance.",
          completed: false,
        },
      ],
    },
    {
      week: "Week 4: April 14 - April 20",
      tasks: [
        {
          task: "Refine the edge drawing step parallelization to increase speedup.",
          completed: false,
        },
        {
          task: "Refine the final color applying step, this part should be straightforward.",
          completed: false,
        },
      ],
    },
    {
      week: "Week 5: April 21 - April 27",
      tasks: [
        {
          task: "Refine the Delaunay triangulation step parallelization to increase speedup to our goal.",
          completed: false,
        },
      ],
    },
    {
      week: "Week 6: April 28 - May 4",
      tasks: [
        {
          task: "Collect statistics, final refinements and complete final report.",
          completed: false,
        },
      ],
    },
  ];

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        flexDirection: "column",
        marginLeft: "15vw",
        marginRight: "15vw",
        marginBottom: "10vh",
      }}
    >
      <Typography
        variant="h5"
        style={{
          color: indigo[500],
          fontWeight: 500,
          textAlign: "left",
          marginBottom: "20px",
        }}
      >
        SCHEDULE
      </Typography>
      <List
        sx={{
          width: "100%",
          margin: "0px",
          padding: "0px",
        }}
      >
        {plan.map((value) => (
          <ListItem
            disableGutters
            sx={{
              width: "100%",
              display: "flex",
              flexDirection: "column",
              margin: "0px",
              padding: "0px",
              marginBottom: "20px",
            }}
          >
            <Typography
              sx={{
                fontFamily: "helvetica",
                fontWeight: 500,
                fontSize: "20px",
                textAlign: "left",
                width: "100%",
                marginBottom: "10px",
              }}
            >
              {value.week}
            </Typography>
            {value.tasks.map((task) => {
              return <TaskItem task={task} />;
            })}
          </ListItem>
        ))}
      </List>
    </div>
  );
}
