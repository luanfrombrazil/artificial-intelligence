
public class NRainhasIndFactory implements IndFactory {

    private final int n;

    public NRainhasIndFactory(int n) {
        this.n = n;
    }

    @Override
    public NRainhasInd getInstance() {
        return new NRainhasInd(this.n);
    }
}
