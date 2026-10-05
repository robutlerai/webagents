class Robutler < Formula
  include Language::Node::Shebang

  desc "Assistant for building and running AI agents, with the webagents command"
  homepage "https://robutler.ai/develop/webagents"
  url "https://registry.npmjs.org/webagents/-/webagents-@VERSION@.tgz"
  sha256 "@SHA256@"
  license "MIT"

  # The Node line the package is tested on. It is keg-only, so `install`
  # points both commands at it.
  depends_on "node@24"

  def install
    system "npm", "install", *std_npm_args
    # `webagents` and `robutler` run on this formula's Node, whichever `node`
    # comes first on PATH.
    rewrite_shebang detected_node_shebang, *libexec.glob("bin/*").map(&:realpath)
    bin.install_symlink libexec.glob("bin/*")
  end

  def caveats
    on_linux do
      <<~EOS
        Shell commands run in a sandbox that uses bubblewrap, socat and ripgrep
        from the system's own package manager, not from Homebrew.
        To check this machine and see what to install:
          webagents sandbox setup
      EOS
    end
  end

  test do
    assert_equal version.to_s, shell_output("#{bin}/webagents --version").strip
    assert_match "Usage:", shell_output("#{bin}/robutler --help")
    system bin/"webagents", "init", "hello"
    assert_path_exists testpath/"hello/AGENT.md"
  end
end
