import Head from 'next/head'
import * as React from 'react';
import Button from '@mui/material/Button';
import SearchBar from './../components/searchbar'
import Paper from '@mui/material/Paper';
import Link from 'next/link';
import BackdropProgress from '../components/backdropload';
import { CirclePacking } from '@nivo/circle-packing'
import Box from '@mui/material/Box';
import Slider from '@mui/material/Slider';
import TextField from '@mui/material/TextField';
import Grid from '@mui/material/Grid';
import { Typography } from '@mui/material';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormHelperText from '@mui/material/FormHelperText';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import { useTheme } from '@mui/material/styles';
import OutlinedInput from '@mui/material/OutlinedInput';
import Chip from '@mui/material/Chip';
import Snackbar from '@mui/material/Snackbar';
import MuiAlert from '@mui/material/Alert';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line as ChartJSLine } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const lineoptions = {
  responsive: true,
  plugins: {
    legend: {
      position: 'top',
    },
    title: {
      display: true,
      text: 'Topic Frequency over Time',
      font: {
        size: 18
      }
    },
  },
  scales: {
    x: {
      title: {
        display: true,
        text: 'Time',
        font: {
          size: 14
        }
      }
    },
    y: {
      title: {
        display: true,
        text: 'Document Frequency',
        font: {
          size: 14
        }
      }
    }
  }
};

function makeColor(colorNum, colors) {
  if (colors < 1) colors = 1;
  // defaults to one color - avoid divide by zero
  return colorNum * (360 / colors) % 360;
}

const Alert = React.forwardRef(function Alert(props, ref) {
  return <MuiAlert elevation={6} ref={ref} variant="filled" {...props} />;
});

const commonProperties = {
  width: 700,
  height: 700,
  padding: 2,
  id: 'name',
  value: 'loc',
  labelsSkipRadius: 16,
}

function getStyles(name, personName, theme) {
  return {
    fontWeight:
      personName.indexOf(name) === -1
        ? theme.typography.fontWeightRegular
        : theme.typography.fontWeightMedium,
  };
}

function parseExractedTopicsAndWeightsResponse(search_query, response) {
  const data = {
    "name": search_query,
    "color": "hsl(44, 70%, 50%)",
    "children": []
  }
  let seen_name_freq = {}
  for (const topic in response) {
    // let main_keyphrase = `${response[topic].main_keyphrase} \n Total Documents: ${response[topic].doc_frequency}`;
    let main_keyphrase = `${response[topic].main_keyphrase}`;
    if (main_keyphrase in seen_name_freq) {
      seen_name_freq[main_keyphrase] += 1
      main_keyphrase += `_${seen_name_freq[main_keyphrase]}`
    } else {
      seen_name_freq[main_keyphrase] = 0
    }
    const nested_data = {
      "name": main_keyphrase,
      "color": "hsl(232, 70%, 50%)",
      "children": []
    }
    for (const keyphrase_index in response[topic].keyphrases) {
      let keyphrase_name = response[topic].keyphrases[keyphrase_index].name
      if (keyphrase_name in seen_name_freq) {
        seen_name_freq[keyphrase_name] += 1
        keyphrase_name += `_${seen_name_freq[keyphrase_name]}`
      } else {
        seen_name_freq[keyphrase_name] = 0
      }
      nested_data["children"].push({
        "name": keyphrase_name,
        "color": "hsl(232, 70%, 50%)",
        "loc": response[topic].keyphrases[keyphrase_index].weight * response[topic].doc_frequency
      })
    }
    data["children"].push(nested_data)
  }
  return data
}

function parseTopicsOverTimeResponse(response) {
  let data = []
  for (const topic in response) {
    let nested_data = {
      "id": `${topic} - ${response[topic].main_keyphrase}`,
      "color": "hsl(53, 70%, 50%)",
      "data": []
    }
    for (const frequency_details of response[topic].frequencies) {
      let frequency_data = {
        "x": frequency_details.timestamp,
        "y": frequency_details.frequency
      }
      nested_data["data"].push(frequency_data)
    }
    data.push(nested_data)
  }
  return data
}

function parseTopicsOverTimeChartJSResponse(response) {
  let labels = [];
  for (const frequency_details of response[Object.keys(response)[0]].frequencies) {
    labels.push(frequency_details.timestamp)
  }
  const responseLength = Object.keys(response).length;
  let datasets = []
  let i = 0
  for (const topic in response) {
    let nested_data = {
      label: `${topic} - ${response[topic].main_keyphrase}`,
      data: [],
      borderColor: `hsl(${makeColor(i, responseLength)}, 100%, 40%)`,
      backgroundColor: `hsl(${makeColor(i, responseLength)}, 100%, 50%)`
    }
    for (const frequency_details of response[topic].frequencies) {
      nested_data["data"].push(frequency_details.frequency)
    }
    datasets.push(nested_data)
    i += 1
  }
  const linedata = {
    labels,
    datasets
  }
  return linedata
}

export default function Home() {
  // const academicDatabases = ["CORE", "arXiv", "Emerald", "ScienceOpen", "Garuda"]
  const academicDatabases = ["CORE", "arXiv", "Emerald", "ScienceOpen"]
  const apiHost = "https://ldaviewerbackend.org"
  const minDistance = 0;
  const currentYear = new Date().getFullYear();
  const theme = useTheme();
  const ITEM_HEIGHT = 48;
  const ITEM_PADDING_TOP = 8;
  const MenuProps = {
    PaperProps: {
      style: {
        maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
        width: 250,
      },
    },
  };

  const [loadingState, setLoadingState] = React.useState(false);
  const [searchQuery, setSearchQuery] = React.useState("");
  const [currentSearchQuery, setCurrentSearchQuery] = React.useState("");
  const [zoomedId, setZoomedId] = React.useState(null);
  const [zoomedDepth, setZoomedDepth] = React.useState(0);
  const [scrapedData, setScrapedData] = React.useState(null);
  const [yearValue, setYearValue] = React.useState([currentYear - 5, currentYear]);
  const [sortBy, setSortBy] = React.useState('relevance');
  const [advancedMenuOpen, setAdvancedMenuOpen] = React.useState(false);
  const [retrievalLimit, setRetrievalLimit] = React.useState(100);
  const [selectedAcademicDatabases, setSelectedAcademicDatabases] = React.useState([]);
  const [searchResultsHistory, setSearchResultsHistory] = React.useState(null)
  const [nInterval, setNInterval] = React.useState(5);

  const [snackbarOpen, setSnackbarOpen] = React.useState(false);
  const [snackbarSeverity, setSnackbarSeverity] = React.useState("success");
  const [snackbarMessage, setSnackbarMessage] = React.useState("");

  async function getSearchResultsHistory() {
    // Set loading state.
    setLoadingState(true)

    // Send the form data to our API and get a response.
    const response = await fetch(`${apiHost}/search_results`, {
      // The method is POST because we are sending data.
      method: 'GET',
    }).then(res => {
      return res
    }).catch(err => {
      setLoadingState(false)
      showSnackbarMessage("Got an error from the server", "error")
      return
    })

    if (typeof response === "undefined") {
      return
    }
    // Get the response data from server as JSON.
    // If server returns the name submitted, that means the form works.
    const result = await response.json()

    setLoadingState(false)

    if (result.status != "success") {
      showSnackbarMessage("Got an error from the server", "error")
      // alert("Got an error:", result.message)
      return
    }

    setSearchResultsHistory(result.results)
  }

  async function viewSpecificSearchResult(id, divide_n = null) {
    // Set loading state.
    setLoadingState(true)

    let url = `https://ldaviewerbackend.org/search_result/${id}`
    if (divide_n != null) {
      url += `?divide_n=${divide_n}`
    }
    // Send the form data to our API and get a response.
    const response = await fetch(url, {
      // The method is POST because we are sending data.
      method: 'GET',
    }).then(res => {
      return res
    }).catch(err => {
      setLoadingState(false)
      showSnackbarMessage("Got an error from the server", "error")
      return
    })

    if (typeof response === "undefined") {
      return
    }

    // Get the response data from server as JSON.
    // If server returns the name submitted, that means the form works.
    const result = await response.json()

    setLoadingState(false)

    if (result.status != "success") {
      // alert("Got an error:", result.message)
      showSnackbarMessage(`Got an error from the server: ${result.message}`, "error")
      return
    }

    result.result.extracted_topics_and_weights = JSON.parse(result.result.extracted_topics_and_weights)
    result.result.extracted_topics_over_time = JSON.parse(result.result.extracted_topics_over_time)
    result.result.metadata = JSON.parse(result.result.metadata)

    showSnackbarMessage("Success!", "success")
    setCurrentSearchQuery(decodeURI(result.result.search_query))
    setScrapedData(result.result)
  }

  async function getLDAResultsFromSearchQuery(searchQuery) {
    // Close advanced menu.
    setAdvancedMenuOpen(false)

    // Set loading state.
    setLoadingState(true)

    // If advanced menu not used, do basic search.
    let data = {}
    if (advancedMenuOpen) {
      selectedAcademicDatabases.forEach((selectedDatabase) => {
        data[`${selectedDatabase.toLowerCase()}_search`] = {
          search_query: searchQuery,
          sort_by: sortBy,
          from_year: yearValue[0],
          to_year: yearValue[1],
          limit: retrievalLimit,
        }
      })
    } else {
      data = {
        core_search: {
          search_query: searchQuery,
          limit: 100,
        },
        arxiv_search: {
          search_query: searchQuery,
          limit: 100,
        },
        emerald_search: {
          search_query: searchQuery,
          limit: 100,
        },
        scienceopen_search: {
          search_query: searchQuery,
          limit: 100,
        },
        // garuda_search: {
        //   search_query: searchQuery,
        //   limit: 100,
        // },
      }
    }

    const JSONdata = JSON.stringify(data)

    // Send the form data to our API and get a response.
    const response = await fetch(`${apiHost}/search`, {
      // Body of the request is the JSON data we created above.
      body: JSONdata,

      // Tell the server we're sending JSON.
      headers: {
        'Content-Type': 'application/json',
      },
      // The method is POST because we are sending data.
      method: 'POST',
    }).then(res => {
      return res
    }).catch(err => {
      setLoadingState(false);
      return
    })

    if (typeof response === "undefined") {
      return
    }

    // Get the response data from server as JSON.
    // If server returns the name submitted, that means the form works.
    const result = await response.json()
    setLoadingState(false);

    if (result.status != "success") {
      showSnackbarMessage(`Got an error from the server: ${result.message}`, "error")
      return
    }

    setCurrentSearchQuery(searchQuery)
    setScrapedData(result)
    getSearchResultsHistory()
  }

  const handleSortByChange = (event) => {
    setSortBy(event.target.value);
  };

  const handleYearChange = (event, newValue, activeThumb) => {
    if (!Array.isArray(newValue)) {
      return;
    }
    if (activeThumb === 0) {
      setYearValue([Math.min(newValue[0], yearValue[1] - minDistance), yearValue[1]]);
    } else {
      setYearValue([yearValue[0], Math.max(newValue[1], yearValue[0] + minDistance)]);
    }
  };

  const handleOnSearchSubmit = async (event) => {
    event.preventDefault()

    getLDAResultsFromSearchQuery(searchQuery)
  }

  const handleAcademicDatabaseChange = (event) => {
    const {
      target: { value },
    } = event;
    setSelectedAcademicDatabases(
      // On autofill we get a stringified value.
      typeof value === 'string' ? value.split(',') : value,
    );
  };

  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }

    setSnackbarOpen(false);
  };

  const showSnackbarMessage = (message, severity) => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setSnackbarOpen(true);
  }

  React.useEffect(() => {
    getSearchResultsHistory()
  }, [])

  return (
    <div className="container">
      <Head>
        <title>LDAViewer</title>
        <link rel="icon" href="/favicon.ico" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap"
        />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/icon?family=Material+Icons"
        />
      </Head>
      <Snackbar open={snackbarOpen} autoHideDuration={3000} onClose={handleSnackbarClose}>
        <Alert onClose={handleSnackbarClose} severity={snackbarSeverity} sx={{ width: '100%' }}>
          {snackbarMessage}
        </Alert>
      </Snackbar>
      <BackdropProgress open={loadingState} message={"Please wait.. Due to the processes running in the background, the processing of queries would usually take around 4 minutes, and would scale linearly with the number of papers to be processed. Please be patient!"}></BackdropProgress>
      <main>
        <h1 className="title">
          Welcome to <Link href="#">LDAViewer!</Link>
        </h1>
        <p className="description">
          Get started by searching your topic of interest.
        </p>
        <Typography align="center" marginLeft={10} marginRight={10} marginBottom={5}>
          <strong>Disclaimer: Due to the processes running in the background, the processing of queries would usually take around 4 minutes, and would scale linearly with the number of papers to be processed. Please be patient!</strong>
        </Typography>
        <SearchBar placeholder={"Search Research Interests"} onSearchChange={setSearchQuery} onButtonClick={() => {
          getLDAResultsFromSearchQuery(searchQuery)
        }} onMenuClick={() => {
          setAdvancedMenuOpen(!advancedMenuOpen)
        }} onSubmit={handleOnSearchSubmit}></SearchBar>
        {advancedMenuOpen &&
          <>
            <Typography variant='h4' mt={5}>Advanced Search</Typography>
            <Grid container spacing={2} marginTop={1} direction="column" alignItems="center" justifyContent="center">
              <Grid item xs={9}>
                <Typography>Search year range from: {yearValue[0]} until {yearValue[1]}</Typography>
                <Slider
                  sx={{ minWidth: 800 }}
                  min={1950}
                  max={currentYear}
                  getAriaLabel={() => 'Year range'}
                  value={yearValue}
                  onChange={handleYearChange}
                  valueLabelDisplay="auto"
                  disableSwap
                  marks={[{ value: 1950, label: '1950' }, { value: currentYear, label: currentYear }]}
                />
              </Grid>
              <Grid item xs={10}>
                <Box display="flex" justifyContent="center">
                  <FormControl sx={{ minWidth: 120, m: 1 }}>
                    <InputLabel id="demo-simple-select-helper-label">Retrieve papers By</InputLabel>
                    <Select
                      labelId="demo-simple-select-helper-label"
                      id="demo-simple-select-helper"
                      value={sortBy}
                      label="Retrieve papers By"
                      onChange={handleSortByChange}
                    >
                      <MenuItem value="relevance">Relevance</MenuItem>
                      <MenuItem value="recent">Newest to Oldest</MenuItem>
                      <MenuItem value="old">Oldest to Newest</MenuItem>
                    </Select>
                    <FormHelperText>Select how you want your papers to be retrieved</FormHelperText>
                  </FormControl>
                  <FormControl sx={{ minWidth: 350, m: 1 }}>
                    <InputLabel id="demo-multiple-chip-label">Selected Academic Database</InputLabel>
                    <Select
                      labelId="demo-multiple-chip-label"
                      id="demo-multiple-chip"
                      multiple
                      value={selectedAcademicDatabases}
                      onChange={handleAcademicDatabaseChange}
                      input={<OutlinedInput id="select-multiple-chip" label="Selected Academic Database" />}
                      renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {selected.map((value) => (
                            <Chip key={value} label={value} />
                          ))}
                        </Box>
                      )}
                      MenuProps={MenuProps}
                    >
                      {academicDatabases.map((name) => (
                        <MenuItem
                          key={name}
                          value={name}
                          style={getStyles(name, selectedAcademicDatabases, theme)}
                        >
                          {name}
                        </MenuItem>
                      ))}
                    </Select>
                    <FormHelperText>Select where your papers would be retrieved from</FormHelperText>
                  </FormControl>
                  <FormControl sx={{ m: 1 }}>
                    <TextField
                      id="standard-number"
                      label="Retrieval Limit per Database"
                      type="number"
                      onChange={(event) => {
                        setRetrievalLimit(event.target.value);
                      }}
                      value={retrievalLimit}
                      min={1}
                      InputLabelProps={{
                        shrink: true,
                      }}
                    />
                  </FormControl>
                </Box>
              </Grid>
            </Grid>
          </>}
        {scrapedData != null &&
          <>
            <Typography variant="h3" mt={2}>LDA results for <strong>{currentSearchQuery}</strong></Typography>
            <Box>
              <Typography variant="h5" mt={2} textAlign={"center"}>Overall Topic Distribution</Typography>
              <CirclePacking
                {...commonProperties}
                margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                data={parseExractedTopicsAndWeightsResponse(currentSearchQuery, scrapedData.extracted_topics_and_weights)}
                enableLabels={true}
                labelsSkipRadius={8}
                labelsFilter={label => label.node.depth <= zoomedDepth + 1}
                labelTextColor={{
                  from: 'color',
                  modifiers: [['darker', 2]],
                }}
                animate={true}
                zoomedId={zoomedId}
                motionConfig="slow"
                onClick={node => {
                  setZoomedId(zoomedId === node.id ? null : node.id)
                  setZoomedDepth(node.depth)
                }}
              />
            </Box>
            <Box>
              <Box style={{ width: 1200, height: 500, margin: "auto" }}>
                <ChartJSLine options={lineoptions} data={parseTopicsOverTimeChartJSResponse(scrapedData.extracted_topics_over_time)} />
              </Box>
              <Box style={{ marginTop: "15vh" }}>
                <FormControl sx={{ m: 1, width: "16vw" }}>
                  <TextField
                    id="standard-number"
                    label="Divide Line Chart into N Contigous Interval"
                    type="number"
                    onChange={(event) => {
                      setNInterval(event.target.value);
                    }}
                    value={nInterval}
                    min={2}
                    InputLabelProps={{
                      shrink: true,
                    }}
                  />
                  <Button onClick={() => {
                    viewSpecificSearchResult(scrapedData.id, nInterval)
                  }}>Change</Button>
                </FormControl>
              </Box>
            </Box>
            {/* <Box style={{ width: 1200, height: 500, margin: "auto" }}>
              <MyResponsiveLine data={parseTopicsOverTimeResponse(scrapedData.extracted_topics_over_time)}></MyResponsiveLine>
            </Box> */}
            <Box >
              <Typography variant="subtitle2">Total Documents Processed: {scrapedData.metadata.total_documents_processed}</Typography>
              <Typography variant="subtitle2">Document Details</Typography>
              {scrapedData.metadata.document_details.map((details) => {
                return (
                  <>
                    <Typography variant="body2">Database: {details.database_name} {details.documents_retrieved} documents</Typography>
                    {/* <Typography variant="body2">Database: {details.database_name}</Typography>
                    // <Typography variant="body2">Documents Retrieved: {details.documents_retrieved}</Typography> */}
                  </>
                )
              })}
              <Typography variant="subtitle2">Number of Topics: {scrapedData.metadata.num_of_topics}</Typography>
              <Typography variant="subtitle2">Coherence Score: {scrapedData.metadata.coherence_score}</Typography>
              <Typography variant="subtitle2">Process Time: {scrapedData.metadata.process_time} minutes</Typography>
              <Button href={`${apiHost}/get_processed_documents/${scrapedData.id}`} variant="contained">
                Download Processed Documents in CSV
              </Button>
            </Box>
          </>
        }
        {searchResultsHistory != null &&
          <>
            <Box alignContent="center" alignItems="center" justifyContent="center">
              <Typography variant='h4' mt={5} align="center">Search Results History</Typography>
              <TableContainer component={Paper} sx={{ m: 2 }}>
                <Table sx={{ minWidth: 650 }} aria-label="simple table">
                  <TableHead>
                    <TableRow>
                      <TableCell>ID</TableCell>
                      <TableCell align="right">Search Query</TableCell>
                      <TableCell align="right">Created At</TableCell>
                      <TableCell align="right">Action</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {searchResultsHistory.map((row) => (
                      <TableRow
                        key={row.id}
                        sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                      >
                        <TableCell component="th" scope="row">
                          {row.id}
                        </TableCell>
                        <TableCell align="right">{row.search_query}</TableCell>
                        <TableCell align="right">{row.create_time}</TableCell>
                        <TableCell align="right"><Button onClick={() => {
                          viewSpecificSearchResult(row.id)
                        }}>View</Button></TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          </>}
        {/* <div className="grid">
          <a href="https://nextjs.org/docs" className="card">
            <h3>Documentation &rarr;</h3>
            <p>Find in-depth information about Next.js features and API.</p>
          </a>

          <a href="https://nextjs.org/learn" className="card">
            <h3>Learn &rarr;</h3>
            <p>Learn about Next.js in an interactive course with quizzes!</p>
          </a>

          <a
            href="https://github.com/vercel/next.js/tree/master/examples"
            className="card"
          >
            <h3>Examples &rarr;</h3>
            <p>Discover and deploy boilerplate example Next.js projects.</p>
          </a>

          <a
            href="https://vercel.com/import?filter=next.js&utm_source=create-next-app&utm_medium=default-template&utm_campaign=create-next-app"
            className="card"
          >
            <h3>Deploy &rarr;</h3>
            <p>
              Instantly deploy your Next.js site to a public URL with Vercel.
            </p>
          </a>
        </div> */}
      </main>

      <footer>
        <a
          href="https://vercel.com?utm_source=create-next-app&utm_medium=default-template&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          A Thesis Project Supported By {' '}<img src="/uiilogo.png" alt="Vercel" className="logo" />
        </a>
      </footer>

      <style jsx>{`
        .container {
          min-height: 100vh;
          padding: 0 0.5rem;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
        }

        main {
          padding: 5rem 0;
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
        }

        footer {
          width: 100%;
          height: 100px;
          border-top: 1px solid #eaeaea;
          display: flex;
          justify-content: center;
          align-items: center;
        }

        footer img {
          margin-left: 0.5rem;
        }

        footer a {
          display: flex;
          justify-content: center;
          align-items: center;
        }

        a {
          color: inherit;
          text-decoration: none;
        }

        .title a {
          color: #0070f3;
          text-decoration: none;
        }

        .title a:hover,
        .title a:focus,
        .title a:active {
          text-decoration: underline;
        }

        .title {
          margin: 0;
          line-height: 1.15;
          font-size: 4rem;
        }

        .title,
        .description {
          text-align: center;
        }

        .description {
          line-height: 1.5;
          font-size: 1.5rem;
        }

        code {
          background: #fafafa;
          border-radius: 5px;
          padding: 0.75rem;
          font-size: 1.1rem;
          font-family: Menlo, Monaco, Lucida Console, Liberation Mono,
            DejaVu Sans Mono, Bitstream Vera Sans Mono, Courier New, monospace;
        }

        .grid {
          display: flex;
          align-items: center;
          justify-content: center;
          flex-wrap: wrap;

          max-width: 800px;
          margin-top: 3rem;
        }

        .card {
          margin: 1rem;
          flex-basis: 45%;
          padding: 1.5rem;
          text-align: left;
          color: inherit;
          text-decoration: none;
          border: 1px solid #eaeaea;
          border-radius: 10px;
          transition: color 0.15s ease, border-color 0.15s ease;
        }

        .card:hover,
        .card:focus,
        .card:active {
          color: #0070f3;
          border-color: #0070f3;
        }

        .card h3 {
          margin: 0 0 1rem 0;
          font-size: 1.5rem;
        }

        .card p {
          margin: 0;
          font-size: 1.25rem;
          line-height: 1.5;
        }

        .logo {
          height: 2em;
        }

        @media (max-width: 600px) {
          .grid {
            width: 100%;
            flex-direction: column;
          }
        }
      `}</style>

      <style jsx global>{`
        html,
        body {
          padding: 0;
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto,
            Oxygen, Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue,
            sans-serif;
        }

        * {
          box-sizing: border-box;
        }
      `}</style>
    </div>
  )
}