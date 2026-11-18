.. _mcp_server:

The index as an MCP server
##########################

The corpus index is not reserved to SPEAR's own sessions. ``spear-mcp`` serves
it over the Model Context Protocol, so any MCP client -- another assistant, an
editor -- searches the same corpora with the same retrieval: dense and lexical
candidates fused by rank, over one corpus or the federation it declares.

.. code-block:: console

   $ docker run --rm -i --entrypoint spear-mcp <image> --corpus <name>

The server speaks on stdin and stdout and exits when its input closes. It
needs no mounted tree and no model endpoint: the image carries the index and
the embedding model a query is embedded with.

Tools
*****

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Tool
     - What it does
   * - ``list_corpora``
     - names the indexed corpora, their size, and the corpora each one is
       searched with
   * - ``search_corpus``
     - ``query`` (required), ``corpus`` and ``top_k``; returns the best chunks,
       each headed by its file path

A path in a result is relative to the root of the corpus that was asked for, a
federated sub-corpus included (``linux/linux/drivers/...``). That is how the
client's own checkout of the same tree names the file, so the client can open
it directly.

``--corpus``, or ``SPEAR_MCP_CORPUS``, names the corpus a search uses when it
does not say. Without it, a search has to name one unless the image holds a
single indexed corpus.

Reaching an image on another machine
************************************

stdio is the only transport, deliberately. The server holds private source,
and a transport that listens on a port would need authentication and
encryption of its own. Over stdio the client starts the server through
whatever already provides both, and ``ssh`` does:

.. code-block:: console

   $ ssh <host> docker run --rm -i --entrypoint spear-mcp <image> --corpus <name>

Registered as the command of a stdio MCP server, that line is the whole
deployment: no port is opened on the host, and who may search is who may log
in.

The first search of a server process loads the embedding model and the index,
which takes some tens of seconds on a CPU and, for a large corpus, several
gigabytes of memory. Later searches of the same process do not pay it again.
